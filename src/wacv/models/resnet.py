from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from src.wacv.models._common import classification_head


def build_resnet50(
    num_attributes: int, *, pretrained: bool, dropout: float
) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = classification_head(  # pyright: ignore[reportAttributeAccessIssue]
        model.fc.in_features, num_attributes, dropout
    )
    return model
