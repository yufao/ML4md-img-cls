import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """Very small CNN用于快速烟雾测试/CPU验证。
    输出: logits (N, num_classes)
    适合小批量快速验证数据管线，不建议用于正式困难样本挖掘。
    """
    def __init__(self, num_classes: int = 8, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


MODEL_REGISTRY = {
    # ResNet 系列: 与 train_kfold.py 中 build_resnet 思路一致，可用于简化快速实验
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    # 轻量自定义模型
    "simple_cnn": SimpleCNN,
}

try:
    # torchvision >= 0.13
    WEIGHTS_REGISTRY = {
        "resnet18": getattr(models, 'ResNet18_Weights', None),
        "resnet50": getattr(models, 'ResNet50_Weights', None),
    }
except Exception:
    WEIGHTS_REGISTRY = {k: None for k in MODEL_REGISTRY.keys()}


def build_classifier(model_name: str = "resnet18", num_classes: int = 8, pretrained: bool = False, dropout: float = 0.2):
    """构建分类模型骨干（单标签或可扩展到多标签）。

    Args:
        model_name: MODEL_REGISTRY 中的键。支持 resnet18/resnet50/simple_cnn。
        num_classes: 输出类别数。
        pretrained: 是否加载 torchvision 预训练权重（需要网络或已缓存）。
        dropout: 在最终 FC 前附加的 Dropout，便于 MC Dropout 不确定性。

    Returns:
        nn.Module: forward(x) -> logits (N, num_classes)

    注意：
    - 若需要 1 通道输入，请在外部将灰度复制到 3 通道或修改第一层 conv。
    - 更完整的 ResNet 输入通道修改、冻结阶段训练等功能在 src/utils/model_utils.build_resnet 中。
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_name {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)

    weights = None
    if pretrained and WEIGHTS_REGISTRY.get(model_name) is not None:
        try:
            weights = WEIGHTS_REGISTRY[model_name].DEFAULT
        except Exception:
            weights = None
    backbone = MODEL_REGISTRY[model_name](weights=weights)
    in_features = backbone.fc.in_features
    layers = []
    if dropout and dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(in_features, num_classes))
    backbone.fc = nn.Sequential(*layers)
    return backbone


class MultiLabelWrapper(nn.Module):
    """Wrapper to provide sigmoid probabilities optionally."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)  # raw logits
