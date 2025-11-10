import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """Very small CNN for smoke tests / CPU quick runs.
    Output: logits (N, num_classes)
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
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
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
    """Build a multi-label classifier backbone.

    Args:
        model_name: key in MODEL_REGISTRY.
        num_classes: number of output labels.
        pretrained: load torchvision pretrained weights (internet may be required).
        dropout: dropout added before final FC for MC Dropout uncertainty.
    Returns:
        nn.Module with attribute .forward(x) -> logits (N, num_classes)
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
