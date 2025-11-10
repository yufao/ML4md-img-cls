import torch
import torch.nn.functional as F
import numpy as np

# Binary metrics

def binarize_logits(logits, prob_th=0.5):
    return (torch.sigmoid(logits) > prob_th).float()

def dice_score_binary(logits, targets, prob_th=0.5, eps=1e-6):
    preds = binarize_logits(logits, prob_th)
    inter = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    dice = (2 * inter + eps) / (union + eps)
    return dice  # (N,)

def iou_score_binary(logits, targets, prob_th=0.5, eps=1e-6):
    preds = binarize_logits(logits, prob_th)
    inter = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou  # (N,)

# Multiclass metrics

@torch.no_grad()
def miou_multiclass(logits, targets_long, num_classes, ignore_index: int=-1, eps=1e-6):
    preds = torch.argmax(logits, dim=1)  # N,H,W
    N, H, W = preds.shape
    miou_list = []
    mdice_list = []
    for n in range(N):
        pred_n = preds[n]
        tgt_n = targets_long[n]
        if ignore_index >= 0:
            valid = tgt_n != ignore_index
            pred_n = pred_n[valid]
            tgt_n = tgt_n[valid]
        if tgt_n.numel() == 0:
            miou_list.append(torch.tensor(float("nan"), device=logits.device))
            mdice_list.append(torch.tensor(float("nan"), device=logits.device))
            continue
        pred_oh = F.one_hot(pred_n, num_classes=num_classes).float()  # [P, C]
        tgt_oh  = F.one_hot(tgt_n.clamp(min=0), num_classes=num_classes).float()
        inter = (pred_oh * tgt_oh).sum(dim=0)                  # C
        pred_area = pred_oh.sum(dim=0)
        tgt_area  = tgt_oh.sum(dim=0)
        union = pred_area + tgt_area - inter + eps
        iou_c = (inter + eps) / union
        present = tgt_area > 0
        miou_list.append(iou_c[present].mean() if present.any() else torch.tensor(float("nan"), device=logits.device))
        dice_c = (2*inter + eps) / (pred_area + tgt_area + eps)
        mdice_list.append(dice_c[present].mean() if present.any() else torch.tensor(float("nan"), device=logits.device))
    return torch.stack(miou_list), torch.stack(mdice_list)  # (N,), (N,)

@torch.no_grad()
def mc_dropout_uncertainty(model, images, num_classes=1, mc_steps=8):
    model.train()  # 打开 dropout
    ent_sum = 0.0
    for _ in range(mc_steps):
        logits = model(images)
        if num_classes <= 1:
            probs = torch.sigmoid(logits)  # N,1,H,W
            p = probs.clamp(1e-6, 1-1e-6)
            ent = -(p*torch.log(p) + (1-p)*torch.log(1-p))     # N,1,H,W
        else:
            probs = torch.softmax(logits, dim=1)               # N,C,H,W
            p = probs.clamp_min(1e-6)
            ent = -(p * torch.log(p)).sum(dim=1, keepdim=True) # N,1,H,W
        ent_sum += ent
    ent_mean = ent_sum / mc_steps
    ent_scalar = ent_mean.mean(dim=(1,2,3))  # (N,)
    model.eval()
    return ent_scalar
