import os
import argparse
import json
import pandas as pd
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from src.models.unet import UNet
from src.data.dataset import SegDataset
from src.utils.losses import bce_dice_loss, ce_dice_loss_multiclass
from src.utils.metrics import dice_score_binary, iou_score_binary, miou_multiclass, mc_dropout_uncertainty
from src.utils.common import set_seed, ensure_dir

def parse_args():
    ap = argparse.ArgumentParser(description="UNet KFold training and hard sample mining")
    ap.add_argument("--manifest", type=str, required=True, help="CSV: id,image_path[,mask_path][,fold]")
    ap.add_argument("--outdir", type=str, default="outputs/run_unet_kfold")
    ap.add_argument("--img-size", type=int, nargs=2, default=[256,256])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-checkpoint", action="store_true")
    ap.add_argument("--prob-th", type=float, default=0.5)
    ap.add_argument("--metric", type=str, choices=["iou","miou","dice","mdice"], default="miou")
    ap.add_argument("--iou-th", type=float, default=0.5, help="hard rule A: 指标低于此")
    ap.add_argument("--hard-percent", type=float, default=0.2, help="hard rule B: 最差百分位")
    ap.add_argument("--mc-steps", type=int, default=0, help=">0 启用 MC Dropout 不确定度")
    ap.add_argument("--num-classes", type=int, default=1, help=">1 走多分类")
    ap.add_argument("--ignore-index", type=int, default=-1, help="忽略标签值，如 255")
    return ap.parse_args()

def train_one_epoch(model, loader, optimizer, device, num_classes, ignore_index):
    model.train()
    total = 0.0
    for batch in loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"]
        logits = model(imgs)
        if num_classes <= 1:
            masks = masks.to(device)
            loss = bce_dice_loss(logits, masks)
        else:
            masks = masks.to(device, dtype=torch.long)
            loss = ce_dice_loss_multiclass(logits, masks, num_classes=num_classes, ignore_index=ignore_index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate_fold(model, loader, device, num_classes, prob_th=0.5, metric="miou", mc_steps=0, ignore_index=-1):
    model.eval()
    ids, dices, ious, mdices, mious, uncs = [], [], [], [], [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"]
        logits = model(imgs)
        if num_classes <= 1:
            masks = masks.to(device)
            d = dice_score_binary(logits, masks, prob_th=prob_th).cpu().numpy()
            i = iou_score_binary(logits, masks, prob_th=prob_th).cpu().numpy()
            ids.extend(batch["id"]); dices.extend(d.tolist()); ious.extend(i.tolist())
            mdices.extend([np.nan]*len(d)); mious.extend([np.nan]*len(d))
        else:
            masks = masks.to(device, dtype=torch.long)
            miou, mdice = miou_multiclass(logits, masks, num_classes=num_classes, ignore_index=ignore_index)
            miou = miou.cpu().numpy(); mdice = mdice.cpu().numpy()
            ids.extend(batch["id"]); mious.extend(miou.tolist()); mdices.extend(mdice.tolist())
            dices.extend([np.nan]*len(miou)); ious.extend([np.nan]*len(miou))
        if mc_steps and mc_steps > 0:
            u = mc_dropout_uncertainty(model, imgs, num_classes=num_classes, mc_steps=mc_steps).cpu().numpy()
        else:
            u = np.full((imgs.size(0),), np.nan, dtype=np.float32)
        uncs.extend(u.tolist())

    df = pd.DataFrame({"id": ids, "dice": dices, "iou": ious, "mdice": mdices, "miou": mious, "uncertainty": uncs})
    use_col = "iou" if (num_classes <= 1 and metric in ["iou","dice"]) else ("miou" if metric in ["miou","mdice"] else "mdice")
    df["metric_used"] = use_col
    df["metric_value"] = df[use_col]
    return df

def pick_hard_samples(df: pd.DataFrame, iou_th: float, hard_percent: float) -> pd.Series:
    rule_a = df["metric_value"] < iou_th
    valid = df["metric_value"].dropna()
    if len(valid) > 0 and 0 < hard_percent < 1:
        k = max(1, int(np.floor(len(valid) * hard_percent)))
        cutoff = valid.nsmallest(k).max()
        rule_b = df["metric_value"] <= cutoff
    else:
        rule_b = pd.Series([False]*len(df), index=df.index)
    return (rule_a.fillna(False) | rule_b.fillna(False))

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.outdir)

    df = pd.read_csv(args.manifest)
    assert "id" in df.columns and "image_path" in df.columns, "manifest 需包含 id,image_path"
    has_mask = ("mask_path" in df.columns) and df["mask_path"].notna().all()
    if not has_mask:
        raise ValueError("需要 mask_path 进行监督训练；无掩膜模式可后续扩展。")

    # 构建折
    if "fold" in df.columns:
        folds_val_idx = [df.index[df["fold"] == f].tolist() for f in sorted(df["fold"].unique())]
        folds = [(df.index.difference(val_idx).tolist(), val_idx) for val_idx in folds_val_idx]
    else:
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        indices = np.arange(len(df))
        folds = [(train_idx.tolist(), val_idx.tolist()) for train_idx, val_idx in kf.split(indices)]

    all_metrics = []
    all_hard_ids = set()
    device = torch.device(args.device)

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        print(f"== Fold {fold_id+1}/{len(folds)}: train {len(train_idx)}, val {len(val_idx)}")
        ds_train = SegDataset(args.manifest, indices=train_idx, img_size=tuple(args.img_size),
                              num_classes=args.num_classes, ignore_index=args.ignore_index)
        ds_val   = SegDataset(args.manifest, indices=val_idx,   img_size=tuple(args.img_size),
                              num_classes=args.num_classes, ignore_index=args.ignore_index)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = UNet(in_channels=3, n_classes=max(1, args.num_classes), base_ch=32, dropout_p=0.1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_monitor = np.inf
        for epoch in range(1, args.epochs+1):
            train_loss = train_one_epoch(model, dl_train, optimizer, device, args.num_classes, args.ignore_index)
            val_df = evaluate_fold(model, dl_val, device, num_classes=args.num_classes,
                                   prob_th=args.prob_th, metric=args.metric, mc_steps=0, ignore_index=args.ignore_index)
            mean_metric = np.nanmean(val_df["metric_value"].values)
            monitor = 1 - (mean_metric if not np.isnan(mean_metric) else 0.0)
            print(f"[fold {fold_id}] epoch {epoch}: train_loss={train_loss:.4f}, {val_df['metric_used'].iloc[0]}={mean_metric:.4f}")
            if monitor < best_monitor:
                best_monitor = monitor
                if args.save_checkpoint:
                    os.makedirs(os.path.join(args.outdir, "checkpoints"), exist_ok=True)
                    ckpt_path = os.path.join(args.outdir, "checkpoints", f"fold{fold_id}_best.pt")
                    torch.save({"state_dict": model.state_dict()}, ckpt_path)

        # 最终评估 + 不确定度
        val_df = evaluate_fold(model, dl_val, device, num_classes=args.num_classes,
                               prob_th=args.prob_th, metric=args.metric, mc_steps=args.mc_steps, ignore_index=args.ignore_index)
        val_df["fold"] = fold_id
        is_hard = pick_hard_samples(val_df, iou_th=args.iou_th, hard_percent=args.hard_percent)
        val_df["is_hard"] = is_hard.astype(bool)
        val_df.to_csv(os.path.join(args.outdir, f"metrics_fold_{fold_id}.csv"), index=False)
        all_metrics.append(val_df)
        all_hard_ids.update(val_df.loc[val_df["is_hard"], "id"].tolist())

    all_df = pd.concat(all_metrics, ignore_index=True)
    all_df.to_csv(os.path.join(args.outdir, "metrics_all.csv"), index=False)
    with open(os.path.join(args.outdir, "hard_sample_ids.txt"), "w") as f:
        for sid in sorted(all_hard_ids):
            f.write(str(sid) + "\n")
    all_df.loc[all_df["is_hard"]].to_csv(os.path.join(args.outdir, "hard_samples.csv"), index=False)

    meta = {
        "manifest": os.path.abspath(args.manifest),
        "kfold": len(folds),
        "epochs": args.epochs,
        "img_size": args.img_size,
        "metric": args.metric,
        "iou_th": args.iou_th,
        "hard_percent": args.hard_percent,
        "mc_steps": args.mc_steps,
        "device": args.device,
        "num_classes": args.num_classes,
        "ignore_index": args.ignore_index,
    }
    with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
