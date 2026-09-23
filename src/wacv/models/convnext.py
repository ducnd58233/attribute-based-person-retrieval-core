from __future__ import annotations

from torch import nn
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_base,
    convnext_tiny,
)

from src.wacv.config import ModelName
from src.wacv.models._common import classification_head


def build_convnext(
    name: ModelName, num_attributes: int, *, pretrained: bool, dropout: float
) -> nn.Module:
    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
    elif name == "convnext_base":
        weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        model = convnext_base(weights=weights)
    else:
        raise ValueError(f"Unsupported ConvNeXt model: {name}")

    output_layer = model.classifier[-1]
    if not isinstance(output_layer, nn.Linear):
        raise TypeError("Unexpected torchvision ConvNeXt classifier layout")
    model.classifier[-1] = classification_head(
        output_layer.in_features, num_attributes, dropout
    )
    return model
