"""使用 ResNet 进行单标签图像分类的 K 折训练脚本，并记录困难样本相关信息。

功能特性：
- 支持基于患者（patient_id）分组的分层 K 折，减少数据泄露风险
- 预热阶段可冻结骨干网络，随后解冻进行完整微调
- 类别权重自动按出现频次的倒数归一化，缓解类别不均衡
- 可选 AMP 混合精度以提升速度与显存效率
- 每折输出验证集预测 CSV（用于后续困难样本挖掘 / 集成预测）
- 可选 JSONL 格式的度量日志追加写入
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
import torch
from torch.amp import autocast

from src.data.cls_dataset import MedicalImageDataset, get_transforms
from src.utils.model_utils import build_resnet, save_checkpoint, load_checkpoint
from src.utils.splits import stratified_group_kfold
from src.utils.logging_utils import setup_logger, JsonlWriter


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_num_classes(meta: pd.DataFrame, label_col: str) -> int:
    return int(meta[label_col].nunique())


def compute_class_weights(labels: np.ndarray, num_classes: int):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.clip(counts, 1.0, None)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def entropy_np(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

# 列别名映射：用于在 CSV 中鲁棒解析不同命名（需与 cls_dataset 同步）
DEFAULT_ALIASES = {
    'image_id': ['image_id', 'id', 'uid', 'img_id', 'filename', 'name'],
    'image_path': ['image_path', 'path', 'file', 'filepath', 'image', 'img_path', 'img'],
    'label': ['label', 'class', 'target', 'y', 'category'],
    'patient_id': ['patient_id', 'patient', 'case_id', 'pid', 'subject', 'group'],
}

def _find_first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def resolve_columns(meta: pd.DataFrame, id_col: str, path_col: str, label_col: str, patient_col: str):
    # 若显式列名不存在则按别名候选顺序尝试匹配
    rid = id_col if id_col in meta.columns else _find_first_col(meta, DEFAULT_ALIASES['image_id'])
    rpath = path_col if path_col in meta.columns else _find_first_col(meta, DEFAULT_ALIASES['image_path'])
    rlab = label_col if label_col in meta.columns else _find_first_col(meta, DEFAULT_ALIASES['label'])
    rpat = None
    if patient_col and patient_col in meta.columns:
        rpat = patient_col
    else:
        rpat = _find_first_col(meta, DEFAULT_ALIASES['patient_id'])
    if not (rid and rpath and rlab):
        missing = []
        if not rid: missing.append('image_id')
        if not rpath: missing.append('image_path')
        if not rlab: missing.append('label')
        raise KeyError(f"Missing required columns {missing}. Available: {list(meta.columns)}")
    return rid, rpath, rlab, rpat


def train_one_epoch(model, loader, optimizer, device, scaler, cls_weight=None):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=scaler is not None):
            logits = model(images)
            if cls_weight is not None:
                loss = F.cross_entropy(logits, labels, weight=cls_weight.to(device))
            else:
                loss = F.cross_entropy(logits, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def record_val_predictions(model, loader, device, out_csv, fold):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc=f"Record fold {fold}"):
        images = batch['image'].to(device)
        labels = batch['label'].cpu().numpy()
        image_ids = batch['image_id']
        patient_ids = batch['patient_id']
        logits = model(images)
        probs = logits.softmax(dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for i in range(len(image_ids)):
            rows.append({
                'image_id': image_ids[i],
                'patient_id': patient_ids[i],
                'true_label': int(labels[i]),
                'pred_label': int(preds[i]),
                'probs': json.dumps(probs[i].tolist()),
                'max_prob': float(probs[i].max()),
                'entropy': entropy_np(probs[i]),
                'fold': int(fold),
                'is_error': int(preds[i] != labels[i]),
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
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--label_col', default='label')
    parser.add_argument('--id_col', default='image_id')
    parser.add_argument('--path_col', default='image_path')
    parser.add_argument('--patient_col', default='patient_id')
    parser.add_argument('--apply_ct_window', action='store_true')
    parser.add_argument('--aug_strategy', default='default', help='Augmentation strategy: default, fundus, cxr, mri')
    parser.add_argument('--freeze_warmup_epochs', type=int, default=2)
    parser.add_argument('--log_jsonl', default='', help='(optional) path to JSONL file to append metrics')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    meta = pd.read_csv(args.meta_csv)
    # 鲁棒解析列名（显式参数 > 别名字典），若缺失关键列则报错
    id_col, path_col, label_col, patient_col = resolve_columns(
        meta,
        id_col=args.id_col,
        path_col=args.path_col,
        label_col=args.label_col,
        patient_col=args.patient_col,
    )
    logger = setup_logger()
    logger.info(f"Columns resolved: id={id_col}, path={path_col}, label={label_col}, patient={patient_col}")
    logger.info(f"Augmentation strategy: {args.aug_strategy}")
    # 将可能的字符串类别编码为整数索引，便于后续模型训练与权重计算
    codes, uniques = pd.factorize(meta[label_col])
    meta['_label_idx'] = codes.astype(int)
    enc_label_col = '_label_idx'
    num_classes = int(meta[enc_label_col].nunique())

    input_mode = 'gray1' if args.pure_gray else 'rgb3'
    in_ch = 1 if args.pure_gray else 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建 K 折：有患者列则做“分组+分层”折分，否则普通 StratifiedKFold
    has_patient = (patient_col is not None) and (patient_col in meta.columns)
    from sklearn.model_selection import StratifiedKFold
    if has_patient:
        folds_iter = stratified_group_kfold(meta, y_col=enc_label_col, group_col=patient_col, n_splits=args.folds, seed=args.seed)
        folds = list(folds_iter)
        # 检查每折验证集是否覆盖全部类别，不足则回退普通 StratifiedKFold
        need_fallback = False
        for _, va_idx in folds:
            uniq = np.unique(meta.iloc[va_idx][enc_label_col].values)
            if len(uniq) < num_classes:
                need_fallback = True
                break
        if need_fallback:
            logger.info("Detected validation folds with missing classes; fallback to StratifiedKFold without patient grouping.")
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
            folds = list(skf.split(meta, meta[enc_label_col].values))
    else:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        folds = list(skf.split(meta, meta[enc_label_col].values))

    logger = setup_logger()
    jsonl = JsonlWriter(args.log_jsonl) if args.log_jsonl else None

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        logger.info(f"==== Fold {fold_idx} ====")
        train_df = meta.iloc[tr_idx].reset_index(drop=True)
        val_df = meta.iloc[va_idx].reset_index(drop=True)

        # 调试：输出当前折训练/验证集中各类别样本数分布
        try:
            tr_cnt = np.bincount(train_df[enc_label_col].values, minlength=num_classes)
            va_cnt = np.bincount(val_df[enc_label_col].values, minlength=num_classes)
            logger.info(f"Fold {fold_idx} label dist train={tr_cnt.tolist()} val={va_cnt.tolist()}")
        except Exception:
            pass

        train_ds = MedicalImageDataset(
            dataframe=train_df,
            images_root=args.img_root,
            transform=get_transforms(args.img_size, True, input_mode, aug_strategy=args.aug_strategy),
            mode=input_mode,
            id_col=id_col,
            path_col=path_col,
            label_col=enc_label_col,
            patient_col=patient_col,
            apply_ct_window=args.apply_ct_window,
            aug_strategy=args.aug_strategy,
        )
        val_ds = MedicalImageDataset(
            dataframe=val_df,
            images_root=args.img_root,
            transform=get_transforms(args.img_size, False, input_mode, aug_strategy=args.aug_strategy),
            mode=input_mode,
            id_col=id_col,
            path_col=path_col,
            label_col=enc_label_col,
            patient_col=patient_col,
            apply_ct_window=args.apply_ct_window,
            aug_strategy=args.aug_strategy,
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
        # 使用新版 torch.amp API 以避免旧接口弃用告警
        scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
        cls_weight = compute_class_weights(train_df[enc_label_col].values, num_classes)

        best_val = float('inf')
        best_path = os.path.join(args.out_dir, f'fold{fold_idx}_best.pth')

        for epoch in range(args.freeze_warmup_epochs):
            tl = train_one_epoch(model, train_loader, optimizer, device, scaler, cls_weight)
            vl, va = evaluate(model, val_loader, device)
            logger.info(f'[Warmup] Fold {fold_idx} Epoch {epoch} train {tl:.4f} val {vl:.4f} acc {va:.4f}')
            if jsonl:
                jsonl.write({'phase': 'warmup', 'fold': fold_idx, 'epoch': epoch, 'train_loss': tl, 'val_loss': vl, 'val_acc': va})
            if vl < best_val:
                best_val = vl
                save_checkpoint(best_path, model, optimizer, epoch, extra={'val_loss': vl, 'acc': va})

        for p in model.parameters():
            p.requires_grad = True

        for epoch in range(args.freeze_warmup_epochs, args.epochs):
            tl = train_one_epoch(model, train_loader, optimizer, device, scaler, cls_weight)
            vl, va = evaluate(model, val_loader, device)
            logger.info(f'Fold {fold_idx} Epoch {epoch} train {tl:.4f} val {vl:.4f} acc {va:.4f}')
            if jsonl:
                jsonl.write({'phase': 'train', 'fold': fold_idx, 'epoch': epoch, 'train_loss': tl, 'val_loss': vl, 'val_acc': va})
            if vl < best_val:
                best_val = vl
                save_checkpoint(best_path, model, optimizer, epoch, extra={'val_loss': vl, 'acc': va})

        load_checkpoint(best_path, model, optimizer=None, map_location=device)
        out_csv = os.path.join(args.out_dir, f'fold{fold_idx}_val_preds.csv')
        record_val_predictions(model, val_loader, device, out_csv, fold_idx)

if __name__ == '__main__':
    main()
