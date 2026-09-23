from __future__ import annotations

from typing import Protocol, cast

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from src.wacv.models._common import classification_head


class _HasFc(Protocol):
    fc: nn.Module


def build_resnet50(
    num_attributes: int, *, pretrained: bool, dropout: float
) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    backbone = cast(_HasFc, model)
    if not isinstance(backbone.fc, nn.Linear):
        raise TypeError("Unexpected ResNet50 fc layout")
    backbone.fc = classification_head(backbone.fc.in_features, num_attributes, dropout)
    return model
