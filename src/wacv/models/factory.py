from __future__ import annotations

from torch import nn

from src.wacv.config import ModelConfig
from src.wacv.models.convnext import build_convnext
from src.wacv.models.resnet import build_resnet50


def build_model(config: ModelConfig, *, num_attributes: int) -> nn.Module:
    """Build one of the explicitly supported attribute classifiers."""
    if config.name == "resnet50":
        return build_resnet50(
            num_attributes, pretrained=config.pretrained, dropout=config.dropout
        )
    return build_convnext(
        config.name,
        num_attributes,
        pretrained=config.pretrained,
        dropout=config.dropout,
    )
