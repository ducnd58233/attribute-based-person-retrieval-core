"""DINOv2 ViT-B/14 and ViT-L/14 with registers as attribute classifiers (252x126).

ViT-B/14 trains with LP-FT; ViT-L/14 keeps its tower frozen and trains LoRA on
every attention output projection plus the head (the E103 recipe).

Weight loading (do not commit checkpoints into git). The architecture always
comes from torch.hub at the pinned commit ``DINOV2_HUB_COMMIT``:

1. **Hub weights (``pretrained=True`` and no ``weights_path``)**: the hub entry
   also downloads the official ``dinov2_vit{b,l}14_reg4_pretrain.pth``.

2. **Local / vendored path (``model.weights_path``)**: hub is called with
   ``pretrained=False`` (use a pre-populated ``$TORCH_HOME/hub`` cache offline),
   then ``load_state_dict`` reads the local ``.pth``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import torch
from torch import nn

from src.wacv.config import MODEL_NATIVE_IMAGE_SIZE
from src.wacv.models._common import (
    build_layerwise_param_groups,
    classification_head,
    set_requires_grad,
)
from src.wacv.models.lora import inject_lora_into_linear

DINOV2_HUB_REPO = "facebookresearch/dinov2"
# Immutable upstream commit (main on 2026-06-03, "Safely load weights from
# specified URLs (#598)"). Pinned so training never runs whatever code the
# default branch holds on the day it is launched.
DINOV2_HUB_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_HUB_ENTRY = "dinov2_vitb14_reg"
DINOV2_L_HUB_ENTRY = "dinov2_vitl14_reg"

_PATCH_SIZE = 14
_TARGET_H, _TARGET_W = MODEL_NATIVE_IMAGE_SIZE["dinov2_vitb14_reg"]


class _HasEmbedDim(Protocol):
    embed_dim: int


class Dinov2AttributeClassifier(nn.Module):
    """DINOv2 vision tower + linear attribute head."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        *,
        feature_dim: int,
        image_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.feature_dim = feature_dim
        self.image_size = image_size
        # Lets HardQueryAdapter infer dim the same way as ConvNeXt / CLIP.
        self.classifier = head

    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        """CLS-token features from DINOv2 ``forward`` (eval path)."""
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Attribute logits."""
        return self.head(self.encode_features(images))

    def freeze_backbone(self) -> None:
        """Freeze the vision tower; keep LoRA adapter weights trainable."""
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = "lora_" in name

    def add_lora_adapters(self, *, rank: int, alpha: float = 16.0) -> int:
        """Add LoRA to every block's attention output projection; return count.

        Base tower weights stay frozen; only LoRA A/B (and the head) train.
        """
        self.freeze_backbone()
        adapted = 0
        for block in self._blocks():
            projection = block.get_submodule("attn.proj")
            if not isinstance(projection, nn.Linear):
                raise TypeError(
                    f"expected attn.proj to be Linear, got {type(projection)}"
                )
            inject_lora_into_linear(projection, rank=rank, alpha=alpha)
            adapted += 1
        return adapted

    def unfreeze_backbone(self) -> None:
        """Unfreeze the vision tower (full fine-tune stage)."""
        set_requires_grad(self.backbone, True)

    def _blocks(self) -> nn.ModuleList:
        blocks = self.backbone.get_submodule("blocks")
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError(f"expected ModuleList of blocks, got {type(blocks)}")
        return blocks

    def layerwise_param_groups(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
        layer_decay: float,
    ) -> list[dict[str, object]]:
        """AdamW param groups with deeper ViT blocks keeping higher LR."""
        return build_layerwise_param_groups(
            head=self.head,
            tower=self.backbone,
            blocks=self._blocks(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            layer_decay=layer_decay,
        )


def load_dinov2_backbone(
    *,
    pretrained: bool,
    weights_path: Path | None = None,
    entry: str = DINOV2_HUB_ENTRY,
) -> nn.Module:
    """Resolve hub ``entry`` (ViT-B/14-reg by default) and optional local weights.

    When ``weights_path`` is set, hub is called with ``pretrained=False`` and
    the file is loaded with ``torch.load`` (CPU map). When ``pretrained`` is
    True and ``weights_path`` is None, hub downloads the FBAI checkpoint.
    """
    load_pretrained = bool(pretrained) and weights_path is None
    backbone = torch.hub.load(
        f"{DINOV2_HUB_REPO}:{DINOV2_HUB_COMMIT}",
        entry,
        pretrained=load_pretrained,
        # torch accepts a bool here; its type stub declares ``str``.
        trust_repo=cast("str", True),
    )
    if not isinstance(backbone, nn.Module):
        raise TypeError(f"unexpected DINOv2 hub return type: {type(backbone)!r}")
    if weights_path is not None:
        payload = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "model" in payload:
            state = payload["model"]
        else:
            state = payload
        if not isinstance(state, dict):
            raise TypeError(
                f"expected state_dict at {weights_path}, got {type(state)!r}"
            )
        backbone.load_state_dict(state, strict=True)
    return backbone


def _attribute_classifier(
    *,
    entry: str,
    num_attributes: int,
    pretrained: bool,
    dropout: float,
    weights_path: Path | None,
    image_size: tuple[int, int],
) -> Dinov2AttributeClassifier:
    """Hub ``entry`` tower plus a linear attribute head.

    Positional embeddings are interpolated at forward time by DINOv2 for any
    H×W divisible by patch size 14 (252×126 → 18×9 patches).
    """
    image_height, image_width = image_size
    if image_height % _PATCH_SIZE != 0 or image_width % _PATCH_SIZE != 0:
        raise ValueError(
            f"image size {(image_height, image_width)} not divisible by {_PATCH_SIZE}"
        )
    backbone = load_dinov2_backbone(
        pretrained=pretrained, weights_path=weights_path, entry=entry
    )
    feature_dim = int(cast(_HasEmbedDim, backbone).embed_dim)
    head = classification_head(feature_dim, num_attributes, dropout)
    return Dinov2AttributeClassifier(
        backbone,
        head,
        feature_dim=feature_dim,
        image_size=(image_height, image_width),
    )


def build_dinov2_vitb14_reg(
    *,
    num_attributes: int,
    pretrained: bool,
    dropout: float,
    weights_path: Path | None = None,
    image_height: int = _TARGET_H,
    image_width: int = _TARGET_W,
) -> Dinov2AttributeClassifier:
    """Build DINOv2 ViT-B/14-reg with a linear attribute head."""
    return _attribute_classifier(
        entry=DINOV2_HUB_ENTRY,
        num_attributes=num_attributes,
        pretrained=pretrained,
        dropout=dropout,
        weights_path=weights_path,
        image_size=(image_height, image_width),
    )


def build_dinov2_vitl14_reg(
    *,
    num_attributes: int,
    pretrained: bool,
    dropout: float,
    weights_path: Path | None = None,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
) -> Dinov2AttributeClassifier:
    """DINOv2 ViT-L/14-reg at 252x126, frozen, with LoRA on every attn.proj."""
    model = _attribute_classifier(
        entry=DINOV2_L_HUB_ENTRY,
        num_attributes=num_attributes,
        pretrained=pretrained,
        dropout=dropout,
        weights_path=weights_path,
        image_size=MODEL_NATIVE_IMAGE_SIZE["dinov2_vitl14_reg"],
    )
    model.add_lora_adapters(rank=lora_rank, alpha=lora_alpha)
    return model
