"""CLEAR-lite hard binary query adapters on top of attribute backbones.

Adapts the hard-query path from Bui et al. CLEAR (arXiv 2403.06119): embed
binary attribute queries and gallery pooled features into a shared space for
ranking, without soft LLM text at inference.
"""

from __future__ import annotations

from typing import Protocol, cast

import torch
import torch.nn.functional as F
from torch import nn


class _ConvNeXtLike(Protocol):
    features: nn.Module
    avgpool: nn.Module
    classifier: nn.Module


class _ResNetLike(Protocol):
    conv1: nn.Module
    bn1: nn.Module
    relu: nn.Module
    maxpool: nn.Module
    layer1: nn.Module
    layer2: nn.Module
    layer3: nn.Module
    layer4: nn.Module
    avgpool: nn.Module
    fc: nn.Module


def _feature_dim_from_classifier(classifier: nn.Module) -> int:
    """Read Linear.in_features from a head that may wrap Dropout."""
    if isinstance(classifier, nn.Linear):
        return int(classifier.in_features)
    if isinstance(classifier, nn.Sequential):
        for module in reversed(list(classifier.children())):
            if isinstance(module, nn.Linear):
                return int(module.in_features)
    raise TypeError("Could not infer feature dim from classifier head")


def pooled_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Extract global pooled features before the attribute classification head."""
    encode = getattr(model, "encode_features", None)
    if callable(encode):
        return cast(torch.Tensor, encode(images))
    if hasattr(model, "features") and hasattr(model, "avgpool"):
        backbone = cast(_ConvNeXtLike, model)
        features = backbone.features(images)
        pooled = backbone.avgpool(features)
        return torch.flatten(pooled, 1)
    if all(hasattr(model, name) for name in ("layer4", "avgpool", "fc")):
        backbone = cast(_ResNetLike, model)
        features = backbone.conv1(images)
        features = backbone.bn1(features)
        features = backbone.relu(features)
        features = backbone.maxpool(features)
        features = backbone.layer1(features)
        features = backbone.layer2(features)
        features = backbone.layer3(features)
        features = backbone.layer4(features)
        pooled = backbone.avgpool(features)
        return torch.flatten(pooled, 1)
    raise TypeError(f"Unsupported backbone for pooled features: {type(model)}")


class HardQueryAdapter(nn.Module):
    """Project binary queries and gallery features into a shared embedding."""

    def __init__(
        self, *, num_attributes: int, feature_dim: int, adapter_dim: int
    ) -> None:
        super().__init__()
        if adapter_dim < 8:
            raise ValueError(f"adapter_dim must be >= 8, got {adapter_dim}")
        self.adapter_dim = adapter_dim
        self.query_proj = nn.Sequential(
            nn.Linear(num_attributes, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim),
        )
        self.gallery_proj = nn.Sequential(
            nn.Linear(feature_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim),
        )
        self.semantic_weight_logits = nn.Parameter(torch.zeros(num_attributes))
        self.margin_scale = nn.Parameter(torch.tensor(0.0))

    def attribute_semantic_weights(self) -> torch.Tensor:
        """Positive per-attribute weights for weighted Hamming (ASMR)."""
        return F.softplus(self.semantic_weight_logits) + 1e-6

    def semantic_margin_matrix(self, query_bits: torch.Tensor) -> torch.Tensor:
        """Pairwise margins proportional to weighted Hamming between queries."""
        from src.wacv.losses import pairwise_weighted_hamming

        scale = F.softplus(self.margin_scale)
        hamming = pairwise_weighted_hamming(
            query_bits, self.attribute_semantic_weights()
        )
        margins = scale * hamming
        margins.fill_diagonal_(0.0)
        return margins

    def encode_queries(self, queries: torch.Tensor) -> torch.Tensor:
        """L2-normalized query embeddings from binary / soft attribute vectors."""
        return F.normalize(self.query_proj(queries.float()), dim=-1)

    def encode_gallery_features(self, features: torch.Tensor) -> torch.Tensor:
        """L2-normalized gallery embeddings from pooled backbone features."""
        return F.normalize(self.gallery_proj(features), dim=-1)


class AttributeModelWithHardQuery(nn.Module):
    """Attribute classifier plus optional CLEAR-lite hard-query adapter."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_attributes: int,
        adapter_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        feature_dim = self._infer_feature_dim(backbone)
        self.hard_query_adapter = HardQueryAdapter(
            num_attributes=num_attributes,
            feature_dim=feature_dim,
            adapter_dim=adapter_dim,
        )

    @staticmethod
    def _infer_feature_dim(backbone: nn.Module) -> int:
        feature_dim_attr = getattr(backbone, "feature_dim", None)
        if isinstance(feature_dim_attr, int) and feature_dim_attr > 0:
            return feature_dim_attr
        if hasattr(backbone, "classifier"):
            classifier = cast(_ConvNeXtLike, backbone).classifier
            head = (
                classifier[-1] if isinstance(classifier, nn.Sequential) else classifier
            )
            return _feature_dim_from_classifier(head)
        if hasattr(backbone, "fc"):
            return _feature_dim_from_classifier(cast(_ResNetLike, backbone).fc)
        raise TypeError(f"Cannot infer feature dim for {type(backbone)}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Attribute logits (same contract as a plain classifier)."""
        return self.backbone(images)

    def gallery_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Encode gallery images into the hard-query embedding space."""
        features = pooled_features(self.backbone, images)
        return self.hard_query_adapter.encode_gallery_features(features)

    def query_embeddings(self, queries: torch.Tensor) -> torch.Tensor:
        """Encode binary query vectors into the hard-query embedding space."""
        return self.hard_query_adapter.encode_queries(queries)


def attach_hard_query_adapter(
    backbone: nn.Module,
    *,
    num_attributes: int,
    adapter_dim: int,
) -> AttributeModelWithHardQuery:
    """Wrap a classifier backbone with a CLEAR-lite hard-query adapter."""
    return AttributeModelWithHardQuery(
        backbone,
        num_attributes=num_attributes,
        adapter_dim=adapter_dim,
    )
