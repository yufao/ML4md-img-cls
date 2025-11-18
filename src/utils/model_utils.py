"""模型与权重工具函数（中文注释）。

包含：
- build_resnet：根据名称/输入通道/是否预训练构建 ResNet
- save_checkpoint / load_checkpoint：保存与加载权重
"""
"""Model utilities for classification backbones (ResNet family).

Highlights:
- build_resnet: Flexible first conv input channels (1 or 3)
- Uses torchvision >= 0.13 weights API safely
- Optional backbone freezing for warm-up training
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

_RESNET_WEIGHTS = {
    'resnet18': models.ResNet18_Weights.DEFAULT,
    'resnet34': models.ResNet34_Weights.DEFAULT,
    'resnet50': models.ResNet50_Weights.DEFAULT,
    'resnet101': models.ResNet101_Weights.DEFAULT,
    'resnet152': models.ResNet152_Weights.DEFAULT,
}

def build_resnet(
    model_name: str = 'resnet50',
    num_classes: int = 2,
    pretrained: bool = True,
    in_ch: int = 3,
    freeze_backbone: bool = False,
):
    if not hasattr(models, model_name):
        raise ValueError(f"Unknown model_name {model_name}")
    weights = _RESNET_WEIGHTS.get(model_name) if pretrained else None
    model_fn = getattr(models, model_name)
    model = model_fn(weights=weights)

    if in_ch != 3:
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_ch, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        if pretrained and in_ch == 1:
            with torch.no_grad():
                # average RGB kernel weights across channel dimension
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        model.conv1 = new_conv

    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError('Expected model to have .fc attribute')

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith('fc'):
                p.requires_grad = False
    return model


def save_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer]=None, epoch: int = 0, extra: Optional[dict]=None):
    ck = {'model_state': model.state_dict(), 'epoch': epoch}
    if optimizer is not None:
        ck['optim_state'] = optimizer.state_dict()
    if extra:
        ck['extra'] = extra
    torch.save(ck, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer]=None, map_location='cpu'):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck['model_state'])
    if optimizer is not None and 'optim_state' in ck:
        optimizer.load_state_dict(ck['optim_state'])
    return ck
