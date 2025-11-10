import torch
import torch.nn.functional as F

# Binary losses

def dice_loss_binary(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2,3)) + eps
    den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps
    dice = num / den
    return 1 - dice.mean()

def bce_dice_loss(logits, targets, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    d = dice_loss_binary(logits, targets)
    return bce_weight * bce + (1 - bce_weight) * d

# Multiclass losses

def soft_dice_loss_multiclass(logits, targets_long, num_classes, ignore_index: int=-1, eps: float=1e-6):
    # logits: (N,C,H,W); targets_long: (N,H,W) long
    probs = torch.softmax(logits, dim=1)
    N, C, H, W = probs.shape
    # build one-hot with ignore mask
    if ignore_index >= 0:
        valid_mask = targets_long != ignore_index
        targets = torch.where(valid_mask, targets_long, torch.zeros_like(targets_long))
    else:
        valid_mask = torch.ones_like(targets_long, dtype=torch.bool)
        targets = targets_long
    one_hot = F.one_hot(targets.clamp(min=0), num_classes=num_classes).permute(0,3,1,2).float()
    one_hot = one_hot * valid_mask.unsqueeze(1)
    probs = probs * valid_mask.unsqueeze(1)
    inter = (probs * one_hot).sum(dim=(2,3)) * 2.0 + eps
    den = probs.sum(dim=(2,3)) + one_hot.sum(dim=(2,3)) + eps
    dice_c = inter / den  # N,C
    gt_present = (one_hot.sum(dim=(2,3)) > 0)  # N,C
    dice_c = torch.where(gt_present, dice_c, torch.nan)
    dice_per_sample = torch.nanmean(dice_c, dim=1)
    return 1 - dice_per_sample.mean()

def ce_dice_loss_multiclass(logits, targets_long, num_classes, ce_weight=0.5, ignore_index: int=-1):
    ce_ii = ignore_index if ignore_index >= 0 else -100
    ce = F.cross_entropy(logits, targets_long, ignore_index=ce_ii)
    d = soft_dice_loss_multiclass(logits, targets_long, num_classes=num_classes, ignore_index=ignore_index)
    return ce_weight * ce + (1 - ce_weight) * d
