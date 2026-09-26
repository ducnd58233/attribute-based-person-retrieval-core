from __future__ import annotations

from torch import nn

from src.wacv.config import ModelConfig
from src.wacv.models.clip import build_clip_attribute_classifier
from src.wacv.models.convnext import build_convnext
from src.wacv.models.resnet import build_resnet50
from src.wacv.query_adapter import attach_hard_query_adapter


def build_model(config: ModelConfig, *, num_attributes: int) -> nn.Module:
    """Build one of the explicitly supported attribute classifiers."""
    if config.name == "resnet50":
        backbone: nn.Module = build_resnet50(
            num_attributes, pretrained=config.pretrained, dropout=config.dropout
        )
    elif config.name == "clip_vit_b16":
        backbone = build_clip_attribute_classifier(
            num_attributes=num_attributes,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )
        if config.freeze_backbone:
            backbone.freeze_backbone()
    else:
        backbone = build_convnext(
            config.name,
            num_attributes,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )
    if not config.hard_query_adapter:
        return backbone
    return attach_hard_query_adapter(
        backbone,
        num_attributes=num_attributes,
        adapter_dim=config.adapter_dim,
    )
