"""多标签分类（如 NIH14）的 ResNet K 折训练脚本。

特性：
- 患者组（patient_id）感知的贪心多标签分层折分，避免数据泄露
- 使用 BCEWithLogitsLoss，并按类别样本不均衡计算 pos_weight
- 支持 AMP 混合精度（torch.amp）
- 每折导出验证集的概率与不确定性（均值熵），便于后续集成与困难样本聚合
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import roc_auc_score, f1_score

from src.data.multilabel_dataset import MultiLabelImageDataset, get_transforms
from src.utils.model_utils import build_resnet, save_checkpoint, load_checkpoint
from src.utils.splits import multilabel_stratified_group_kfold
from src.utils.logging_utils import setup_logger, JsonlWriter


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(train_df: pd.DataFrame, label_cols):
    y = train_df[label_cols].values.astype(np.float32)
    pos = y.sum(axis=0)
    neg = (y.shape[0] - pos)
    pw = np.divide(neg, np.clip(pos, 1.0, None))
    pw = pw / np.maximum(pw.mean(), 1e-6)
    return torch.tensor(pw, dtype=torch.float32)


def entropy_np_binary(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@torch.no_grad()
def evaluate(model, loader, device, prob_th=0.5):
    """评估宏 AUC 与 micro-F1。若某类正/负全为 0，则跳过该类的 AUC。"""
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['labels'].cpu().numpy()
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true.append(labels)
        y_prob.append(probs)
    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    # macro AUC (ignore classes with only one label)
    auc_list = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        yp = y_prob[:, c]
        if yt.sum() == 0 or (1 - yt).sum() == 0:
            continue
        try:
            auc_list.append(roc_auc_score(yt, yp))
        except Exception:
            continue
    macro_auc = float(np.mean(auc_list)) if auc_list else float('nan')
    y_pred = (y_prob > prob_th).astype(np.float32)
    micro_f1 = float(f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0))
    return macro_auc, micro_f1


@torch.no_grad()
def record_val_predictions(model, loader, device, out_csv, fold):
    """保存验证集逐图预测、均值熵与元信息，供后续分析。"""
    model.eval()
    rows = []
    for batch in tqdm(loader, desc=f"Record fold {fold}"):
        images = batch['image'].to(device)
        labels = batch['labels'].cpu().numpy()
        image_ids = batch['image_id']
        patient_ids = batch['patient_id']
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        ent = entropy_np_binary(probs).mean(axis=1)  # per-sample mean entropy across classes
        for i in range(len(image_ids)):
            rows.append({
                'image_id': image_ids[i],
                'patient_id': patient_ids[i],
                'true_labels': json.dumps(labels[i].tolist()),
                'probs': json.dumps(probs[i].tolist()),
                'mean_entropy': float(ent[i]),
                'fold': int(fold),
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_csv', required=True)
    parser.add_argument('--img_root', default='')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pure_gray', action='store_true')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--id_col', default='image_id')
    parser.add_argument('--path_col', default='image_path')
    parser.add_argument('--patient_col', default='patient_id')
    parser.add_argument('--label_cols', default='', help='Comma-separated list of label column names; if empty, auto-detect NIH14 columns')
    parser.add_argument('--labels_json_col', default='', help='Alternative labels JSON column name')
    parser.add_argument('--freeze_warmup_epochs', type=int, default=1)
    parser.add_argument('--log_jsonl', default='', help='(optional) path to JSONL to append metrics')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    meta = pd.read_csv(args.meta_csv)

    # Resolve label columns
    label_cols = [s.strip() for s in args.label_cols.split(',') if s.strip()]
    labels_json_col = args.labels_json_col or None
    # If no explicit label cols, try NIH14 default columns
    if not label_cols:
        nih_cols = [
            'Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax',
            'Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia'
        ]
        label_cols = [c for c in nih_cols if c in meta.columns]
    num_classes = len(label_cols) if label_cols else None

    input_mode = 'gray1' if args.pure_gray else 'rgb3'
    in_ch = 1 if args.pure_gray else 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    has_patient = (args.patient_col in meta.columns)
    if not has_patient:
        # Create synthetic group id to avoid empty string mixing
        meta['_group'] = meta.index.astype(str)
        patient_col = '_group'
    else:
        patient_col = args.patient_col

    # Build folds via multilabel group stratification
    if label_cols:
        folds = list(multilabel_stratified_group_kfold(meta, y_cols=label_cols, group_col=patient_col, n_splits=args.folds, seed=args.seed))
    else:
        # labels_json cannot be aggregated easily without decoding; decode once
        labs = meta[labels_json_col].apply(lambda x: np.array(json.loads(x), dtype=np.float32))
        y_mat = np.stack(labs.values.tolist())
        tmp_df = meta.copy()
        tmp_cols = [f'lbl_{i}' for i in range(y_mat.shape[1])]
        tmp_df[tmp_cols] = y_mat
        folds = list(multilabel_stratified_group_kfold(tmp_df, y_cols=tmp_cols, group_col=patient_col, n_splits=args.folds, seed=args.seed))
        label_cols = tmp_cols
        num_classes = y_mat.shape[1]

    logger = setup_logger()
    jsonl = JsonlWriter(args.log_jsonl) if args.log_jsonl else None

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        logger.info(f"==== Fold {fold_idx} ====")
        train_df = meta.iloc[tr_idx].reset_index(drop=True)
        val_df = meta.iloc[va_idx].reset_index(drop=True)

        train_ds = MultiLabelImageDataset(
            dataframe=train_df,
            images_root=args.img_root,
            transform=get_transforms(args.img_size, True, input_mode),
            mode=input_mode,
            id_col=args.id_col,
            path_col=args.path_col,
            patient_col=args.patient_col if has_patient else '_group',
            label_cols=label_cols if all(c in train_df.columns for c in label_cols) else None,
            labels_json_col=labels_json_col,
        )
        val_ds = MultiLabelImageDataset(
            dataframe=val_df,
            images_root=args.img_root,
            transform=get_transforms(args.img_size, False, input_mode),
            mode=input_mode,
            id_col=args.id_col,
            path_col=args.path_col,
            patient_col=args.patient_col if has_patient else '_group',
            label_cols=label_cols if all(c in val_df.columns for c in label_cols) else None,
            labels_json_col=labels_json_col,
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        model = build_resnet(
            model_name=args.model,
            num_classes=num_classes,
            pretrained=args.pretrained,
            in_ch=in_ch,
            freeze_backbone=(args.freeze_warmup_epochs > 0),
        ).to(device)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
        pos_weight = compute_pos_weight(train_df, label_cols).to(device)

        best_metric = -1e9
        best_path = os.path.join(args.out_dir, f'fold{fold_idx}_best.pth')

        def train_one_epoch(loader):
            model.train()
            total_loss = 0.0
            for batch in loader:
                images = batch['image'].to(device)
                labels = batch['labels'].to(device)
                optimizer.zero_grad(set_to_none=True)
                with autocast('cuda', enabled=scaler is not None):
                    logits = model(images)
                    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward(); optimizer.step()
                total_loss += loss.item() * images.size(0)
            return total_loss / len(loader.dataset)

        for epoch in range(args.freeze_warmup_epochs):
            tl = train_one_epoch(train_loader)
            macro_auc, micro_f1 = evaluate(model, val_loader, device)
            logger.info(f'[Warmup] Fold {fold_idx} Epoch {epoch} train {tl:.4f} val_auc {macro_auc:.4f} micro_f1 {micro_f1:.4f}')
            if jsonl:
                jsonl.write({'phase': 'warmup', 'fold': fold_idx, 'epoch': epoch, 'train_loss': tl, 'val_auc': macro_auc, 'micro_f1': micro_f1})
            if macro_auc > best_metric:
                best_metric = macro_auc
                save_checkpoint(best_path, model, optimizer, epoch, extra={'val_auc': macro_auc, 'micro_f1': micro_f1})

        for p in model.parameters():
            p.requires_grad = True

        for epoch in range(args.freeze_warmup_epochs, args.epochs):
            tl = train_one_epoch(train_loader)
            macro_auc, micro_f1 = evaluate(model, val_loader, device)
            logger.info(f'Fold {fold_idx} Epoch {epoch} train {tl:.4f} val_auc {macro_auc:.4f} micro_f1 {micro_f1:.4f}')
            if jsonl:
                jsonl.write({'phase': 'train', 'fold': fold_idx, 'epoch': epoch, 'train_loss': tl, 'val_auc': macro_auc, 'micro_f1': micro_f1})
            if macro_auc > best_metric:
                best_metric = macro_auc
                save_checkpoint(best_path, model, optimizer, epoch, extra={'val_auc': macro_auc, 'micro_f1': micro_f1})

        load_checkpoint(best_path, model, optimizer=None, map_location=device)
        out_csv = os.path.join(args.out_dir, f'fold{fold_idx}_val_preds.csv')
        record_val_predictions(model, val_loader, device, out_csv, fold_idx)


if __name__ == '__main__':
    main()
