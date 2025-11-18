"""Ensemble prediction using all fold checkpoints to derive disagreement metrics.

Generates per-image:
    avg_probs, avg_max_prob, avg_entropy, unique_preds (number of distinct model argmax labels).

CLI example:
    python -m src.scripts.predict_ensemble \
        --meta_csv data/meta.csv \
        --img_root data/images \
        --ck_dir outputs/ord5k_cls \
        --out_csv outputs/ord5k_cls/ensemble_preds.csv \
        --pretrained
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

from src.data.cls_dataset import MedicalImageDataset, get_transforms
from src.utils.model_utils import build_resnet, load_checkpoint

DEFAULT_ALIASES = {
    'image_id': ['image_id', 'id', 'uid', 'img_id', 'filename', 'name'],
    'image_path': ['image_path', 'path', 'file', 'filepath', 'image', 'img_path', 'img'],
    'label': ['label', 'class', 'target', 'y', 'category'],
    'patient_id': ['patient_id', 'patient', 'case_id', 'pid', 'subject', 'group'],
}

def _find_first_col(df: pd.DataFrame, key: str, explicit: str | None):
    if explicit and explicit in df.columns:
        return explicit
    for c in DEFAULT_ALIASES.get(key, []):
        if c in df.columns:
            return c
    raise KeyError(f"Missing required column for {key}. Available: {list(df.columns)}")

@torch.no_grad()
def predict_with_models(models, loader, device):
    for m in models: m.eval()
    rows = []
    for batch in tqdm(loader, desc='Predict ensemble'):
        images = batch['image'].to(device)
        image_ids = batch['image_id']
        patient_ids = batch['patient_id']
        probs_list = []
        for m in models:
            logits = m(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
        probs_arr = np.stack(probs_list, axis=1)  # (B,K,C)
        avg_probs = probs_arr.mean(axis=1)
        preds = avg_probs.argmax(axis=1)
        max_probs = avg_probs.max(axis=1)
        model_preds = probs_arr.argmax(axis=2)  # (B,K)
        unique_counts = np.array([len(set(row)) for row in model_preds])
        ent = -(avg_probs * np.clip(np.log(avg_probs + 1e-12), -1e6, 1e6)).sum(axis=1)
        for i in range(len(image_ids)):
            rows.append({
                'image_id': image_ids[i],
                'patient_id': patient_ids[i],
                'avg_probs': json.dumps(avg_probs[i].tolist()),
                'avg_max_prob': float(max_probs[i]),
                'avg_entropy': float(ent[i]),
                'unique_preds': int(unique_counts[i]),
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
    # Optional explicit column names (falls back to aliases if missing)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--id_col', default=None)
    parser.add_argument('--path_col', default=None)
    parser.add_argument('--patient_col', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta = pd.read_csv(args.meta_csv)
    # Resolve columns robustly
    id_col = _find_first_col(meta, 'image_id', args.id_col)
    path_col = _find_first_col(meta, 'image_path', args.path_col)
    try:
        patient_col = _find_first_col(meta, 'patient_id', args.patient_col)
    except KeyError:
        patient_col = None
    label_col = _find_first_col(meta, 'label', args.label_col)
    num_classes = int(meta[label_col].nunique())
    input_mode = 'gray1' if args.pure_gray else 'rgb3'
    in_ch = 1 if args.pure_gray else 3

    ds = MedicalImageDataset(
        dataframe=meta,
        images_root=args.img_root,
        transform=get_transforms(args.img_size, False, input_mode),
        mode=input_mode,
        id_col=id_col,
        path_col=path_col,
        label_col=label_col,
        patient_col=patient_col,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    ck_paths = sorted(glob.glob(os.path.join(args.ck_dir, 'fold*_best.pth')))
    assert ck_paths, f'No *_best.pth in {args.ck_dir}'
    models = []
    for p in ck_paths:
        m = build_resnet(args.model, num_classes=num_classes, pretrained=args.pretrained, in_ch=in_ch)
        load_checkpoint(p, m, optimizer=None, map_location=device)
        m.to(device)
        models.append(m)

    df = predict_with_models(models, loader, device)
    df.to_csv(args.out_csv, index=False)
    print(f'Wrote {args.out_csv} using {len(ck_paths)} models')

if __name__ == '__main__':
    main()
