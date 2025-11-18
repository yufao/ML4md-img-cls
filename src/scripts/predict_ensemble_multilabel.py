"""多标签分类模型的集成推理脚本（multilabel）。

按图像输出以下不确定性与一致性指标：
- avg_probs：各类别在不同折/模型上的概率求均值
- mean_entropy：二元熵在类别维度的均值，越大不确定性越高
- disagreement_jaccard：1 − 平均 Jaccard（对各折二值化结果做成对 Jaccard），越大分歧越高
"""
import os
import glob
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.data.multilabel_dataset import MultiLabelImageDataset, get_transforms
from src.utils.model_utils import build_resnet, load_checkpoint


def binary_entropy(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def jaccard(a, b):
    inter = (a & b).sum(axis=1)
    union = (a | b).sum(axis=1) + 1e-6
    return inter / union


@torch.no_grad()
def predict_with_models(models, loader, device, prob_th=0.5):
    """用若干模型对同一批图像做集成推理，计算平均概率/熵/分歧。

    参数：
    - models: 已加载权重的模型列表
    - loader: 验证/全量数据的 DataLoader
    - device: 运行设备
    - prob_th: 二值化阈值（用于 Jaccard 分歧）
    """
    for m in models: m.eval()
    rows = []
    for batch in tqdm(loader, desc='Predict ensemble (multilabel)'):
        images = batch['image'].to(device)
        image_ids = batch['image_id']
        patient_ids = batch['patient_id']
        probs_list = []
        for m in models:
            logits = m(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
        probs_arr = np.stack(probs_list, axis=1)  # (B,K,C)
        avg_probs = probs_arr.mean(axis=1)        # (B,C)
        mean_ent = binary_entropy(avg_probs).mean(axis=1)

        # Disagreement via Jaccard across folds
        k = probs_arr.shape[1]
        bin_preds = (probs_arr > prob_th).astype(np.int32)  # (B,K,C)
        # compute average pairwise Jaccard
        if k > 1:
            pairs = []
            for i in range(k):
                for j in range(i+1, k):
                    p_i = bin_preds[:, i, :]
                    p_j = bin_preds[:, j, :]
                    jac = jaccard(p_i, p_j)
                    pairs.append(jac)
            mean_jac = np.mean(np.stack(pairs, axis=1), axis=1)
            dis_jac = 1.0 - mean_jac
        else:
            dis_jac = np.zeros(avg_probs.shape[0], dtype=np.float32)

        for i in range(len(image_ids)):
            rows.append({
                'image_id': image_ids[i],
                'patient_id': patient_ids[i],
                'avg_probs': json.dumps(avg_probs[i].tolist()),
                'mean_entropy': float(mean_ent[i]),
                'disagreement_jaccard': float(dis_jac[i]),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_csv', required=True)
    parser.add_argument('--img_root', default='')
    parser.add_argument('--ck_dir', required=True)
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pure_gray', action='store_true')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--id_col', default='image_id')
    parser.add_argument('--path_col', default='image_path')
    parser.add_argument('--patient_col', default='patient_id')
    parser.add_argument('--label_cols', default='')
    parser.add_argument('--labels_json_col', default='')
    parser.add_argument('--prob_th', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta = pd.read_csv(args.meta_csv)

    label_cols = [s.strip() for s in args.label_cols.split(',') if s.strip()]
    labels_json_col = args.labels_json_col or None
    if not label_cols:
        nih_cols = [
            'Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax',
            'Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia'
        ]
        label_cols = [c for c in nih_cols if c in meta.columns]
    num_classes = len(label_cols) if label_cols else None

    input_mode = 'gray1' if args.pure_gray else 'rgb3'
    in_ch = 1 if args.pure_gray else 3

    ds = MultiLabelImageDataset(
        dataframe=meta,
        images_root=args.img_root,
        transform=get_transforms(args.img_size, False, input_mode),
        mode=input_mode,
        id_col=args.id_col,
        path_col=args.path_col,
        patient_col=args.patient_col if args.patient_col in meta.columns else None,
        label_cols=label_cols if label_cols else None,
        labels_json_col=labels_json_col,
    )
    if num_classes is None:
        num_classes = ds.num_classes

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    ck_paths = sorted(glob.glob(os.path.join(args.ck_dir, 'fold*_best.pth')))
    assert ck_paths, f'No *_best.pth in {args.ck_dir}'
    models = []
    for p in ck_paths:
        m = build_resnet(args.model, num_classes=num_classes, pretrained=args.pretrained, in_ch=in_ch)
        load_checkpoint(p, m, optimizer=None, map_location=device)
        m.to(device)
        models.append(m)

    df = predict_with_models(models, loader, device, prob_th=args.prob_th)
    df.to_csv(args.out_csv, index=False)
    print(f'Wrote {args.out_csv} using {len(ck_paths)} models')


if __name__ == '__main__':
    main()
