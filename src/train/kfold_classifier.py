import argparse
import os
import sys
import json
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold

# ensure project root in path if script executed directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.classifier import build_classifier


class FundusClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_root, row['filename'])
        from PIL import Image
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label_vec = torch.tensor(row['target_vec'], dtype=torch.float32)
        return img, label_vec, row['sample_id']


def parse_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # keep rows with filename and target
    # parse target string like "[1, 0, 0, 0, 0, 0, 0, 0]"
    parsed = []
    for _, r in df.iterrows():
        if isinstance(r.get('filename'), str) and isinstance(r.get('target'), str):
            try:
                vec = json.loads(r['target'])
                if len(vec) == 8:
                    parsed.append({
                        'sample_id': r['ID'],
                        'filename': r['filename'],
                        'target_vec': vec
                    })
            except Exception:
                continue
    return pd.DataFrame(parsed)


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    eps = 1e-8
    tp = (y_pred * y_true).sum(axis=0)
    fp = (y_pred * (1 - y_true)).sum(axis=0)
    fn = ((1 - y_pred) * y_true).sum(axis=0)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    macro_f1 = f1.mean()
    return {
        'macro_f1': float(macro_f1),
        'per_class_f1': f1.tolist(),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist()
    }


def mc_dropout_uncertainty(model: nn.Module, inputs: torch.Tensor, samples: int = 10) -> torch.Tensor:
    model.train()  # activate dropout
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds.append(probs.unsqueeze(0))
    stacked = torch.cat(preds, dim=0)  # (S, B, C)
    mean_p = stacked.mean(dim=0)
    # entropy per sample: -sum(p*log(p)+(1-p)*log(1-p)) averaged over classes
    entropy = -(mean_p * (mean_p.clamp_min(1e-8).log()) + (1 - mean_p) * ((1 - mean_p).clamp_min(1e-8).log()))
    return entropy.mean(dim=1)  # (B,)


def mine_hard_samples(sample_ids: List, y_true: np.ndarray, y_prob: np.ndarray, uncertainty: np.ndarray,
                      conf_threshold: float = 0.6, bottom_percent: float = 10.0) -> List[dict]:
    y_pred = (y_prob >= 0.5).astype(int)
    reasons = []
    # per-sample correctness (all labels match)
    incorrect = (y_pred != y_true).any(axis=1)
    # low confidence: any true positive label prob < conf_threshold
    low_conf = []
    for i in range(len(y_true)):
        true_pos_indices = np.where(y_true[i] == 1)[0]
        if len(true_pos_indices) == 0:
            low_conf.append(False)
        else:
            vals = y_prob[i, true_pos_indices]
            low_conf.append(np.any(vals < conf_threshold))
    low_conf = np.array(low_conf)
    # uncertainty ranking
    n = len(sample_ids)
    k = max(1, int(n * bottom_percent / 100.0))
    unc_indices = np.argsort(-uncertainty)[:k]  # highest uncertainty
    unc_mask = np.zeros(n, dtype=bool)
    unc_mask[unc_indices] = True

    for i, sid in enumerate(sample_ids):
        r = []
        if incorrect[i]:
            r.append('incorrect')
        if low_conf[i]:
            r.append('low_conf')
        if unc_mask[i]:
            r.append('high_uncertainty')
        if r:
            reasons.append({
                'sample_id': int(sid),
                'reasons': ';'.join(r),
                'uncertainty': float(uncertainty[i]),
                'pred_probs': y_prob[i].tolist(),
                'gt': y_true[i].tolist()
            })
    return reasons


def run_fold(train_df, val_df, args, device, fold_idx: int, out_dir: str):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_ds = FundusClsDataset(train_df, args.image_root, transform)
    val_ds = FundusClsDataset(val_df, args.image_root, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_classifier(args.model, num_classes=8, pretrained=False, dropout=args.dropout)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # validation
    model.eval()
    all_probs = []
    all_gts = []
    all_ids = []
    with torch.no_grad():
        for imgs, labels, sids in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_gts.append(labels.numpy())
            all_ids.extend(sids)
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_gts, axis=0)
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)

    # uncertainty (MC Dropout) on validation set
    unc_probs = []
    with torch.no_grad():
        for imgs, _, _ in val_loader:
            imgs = imgs.to(device)
            unc = mc_dropout_uncertainty(model, imgs, samples=args.mc_samples)
            unc_probs.append(unc.cpu().numpy())
    uncertainty = np.concatenate(unc_probs, axis=0)

    hard = mine_hard_samples(all_ids, y_true, y_prob, uncertainty,
                             conf_threshold=args.hard_threshold,
                             bottom_percent=args.hard_percentile)

    fold_dir = os.path.join(out_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    with open(os.path.join(fold_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(hard).to_csv(os.path.join(fold_dir, 'hard_samples.csv'), index=False)

    return metrics, hard


def main():
    parser = argparse.ArgumentParser(description="KFold multi-label fundus classifier with hard sample mining")
    parser.add_argument('--csv', type=str, required=True, help='Path to full_df.csv')
    parser.add_argument('--image-root', type=str, required=True, help='Directory of images (preprocessed_images)')
    parser.add_argument('--out-dir', type=str, default='outputs/ord5k_cls', help='Output directory')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--hard-threshold', type=float, default=0.6)
    parser.add_argument('--hard-percentile', type=float, default=10.0)
    parser.add_argument('--mc-samples', type=int, default=10)
    parser.add_argument('--limit', type=int, default=0, help='Limit number of samples for quick runs')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = parse_csv(args.csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    if df.empty:
        raise RuntimeError("Parsed dataframe is empty. Check csv format.")

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    all_metrics = []
    all_hard = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        metrics, hard = run_fold(train_df, val_df, args, device, fold_idx, args.out_dir)
        all_metrics.append({'fold': fold_idx, **metrics})
        for h in hard:
            h['fold'] = fold_idx
            all_hard.append(h)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'summary_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    pd.DataFrame(all_hard).to_csv(os.path.join(args.out_dir, 'all_hard_samples.csv'), index=False)

    print("Done. Metrics and hard samples saved.")


if __name__ == '__main__':
    main()
