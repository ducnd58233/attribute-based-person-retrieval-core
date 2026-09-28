from __future__ import annotations

from torch import nn
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_base,
    convnext_tiny,
)

from src.wacv.config import MixStyleConfig, ModelName
from src.wacv.models._common import classification_head
from src.wacv.models.mixstyle import MixStyle

# Torchvision ConvNeXt ``features`` layout: stem, stage1, down, stage2, down, ...
_STAGE_END_INDEX: dict[int, int] = {1: 1, 2: 3, 3: 5, 4: 7}


def _inject_mixstyle(features: nn.Sequential, config: MixStyleConfig) -> nn.Sequential:
    """Insert MixStyle after the configured ConvNeXt stages (1-based)."""
    unknown = sorted(set(config.stages) - set(_STAGE_END_INDEX))
    if unknown:
        raise ValueError(
            f"Unsupported MixStyle stages {unknown}; expected subset of "
            f"{sorted(_STAGE_END_INDEX)}"
        )
    end_indices = {_STAGE_END_INDEX[stage] for stage in config.stages}
    modules: list[nn.Module] = []
    for index, layer in enumerate(features):
        modules.append(layer)
        if index in end_indices:
            modules.append(MixStyle(p=config.p, alpha=float(config.alpha)))
    return nn.Sequential(*modules)


def build_convnext(
    name: ModelName,
    num_attributes: int,
    *,
    pretrained: bool,
    dropout: float,
    mixstyle: MixStyleConfig | None = None,
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
    if mixstyle is not None and mixstyle.enabled:
        model.features = _inject_mixstyle(model.features, mixstyle)
    return model
