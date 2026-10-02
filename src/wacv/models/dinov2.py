"""DINOv2 ViT-B/14 with registers as an attribute classifier (252x126 LP-FT).

Weight loading (do not commit checkpoints into git):

1. **torch.hub (default when ``pretrained=True`` and no ``weights_path``)**
   ``torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")``
   pulls architecture +
   ``dinov2_vitb14_reg4_pretrain.pth`` from
   ``https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/``.

2. **Local / vendored path (``model.weights_path``)**
   Architecture still comes from torch.hub with ``pretrained=False`` (use a
   pre-populated ``$TORCH_HOME/hub`` cache offline). Then
   ``load_state_dict`` from the local ``.pth`` — the path to vendor into a
   training sandbox or later Codabench zip packaging.

3. **Hugging Face (manual convert)**
   ``facebook/dinov2-with-registers-base`` — not loaded automatically; export
   a torch.hub-compatible state_dict if you prefer HF as the download source.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.wacv.models._common import (
    build_layerwise_param_groups,
    classification_head,
    set_requires_grad,
)

DINOV2_HUB_REPO = "facebookresearch/dinov2"
DINOV2_HUB_ENTRY = "dinov2_vitb14_reg"

_PATCH_SIZE = 14
_TARGET_H = 252
_TARGET_W = 126
_FEATURE_DIM = 768


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
        """Freeze the vision tower (linear-probe stage)."""
        set_requires_grad(self.backbone, False)

    def unfreeze_backbone(self) -> None:
        """Unfreeze the vision tower (full fine-tune stage)."""
        set_requires_grad(self.backbone, True)

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
            blocks=self.backbone.blocks,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            layer_decay=layer_decay,
        )


def load_dinov2_vitb14_reg_backbone(
    *,
    pretrained: bool,
    weights_path: Path | None = None,
) -> nn.Module:
    """Resolve ``dinov2_vitb14_reg`` via torch.hub and optional local weights.

    When ``weights_path`` is set, hub is called with ``pretrained=False`` and
    the file is loaded with ``torch.load`` (CPU map). When ``pretrained`` is
    True and ``weights_path`` is None, hub downloads the FBAI checkpoint.
    """
    load_pretrained = bool(pretrained) and weights_path is None
    backbone = torch.hub.load(
        DINOV2_HUB_REPO,
        DINOV2_HUB_ENTRY,
        pretrained=load_pretrained,
        trust_repo=True,
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


def build_dinov2_vitb14_reg(
    *,
    num_attributes: int,
    pretrained: bool,
    dropout: float,
    weights_path: Path | None = None,
    image_height: int = _TARGET_H,
    image_width: int = _TARGET_W,
) -> Dinov2AttributeClassifier:
    """Build DINOv2 ViT-B/14-reg with a linear attribute head.

    Positional embeddings are interpolated at forward time by DINOv2 for any
    H×W divisible by patch size 14 (252×126 → 18×9 patches).
    """
    if image_height % _PATCH_SIZE != 0 or image_width % _PATCH_SIZE != 0:
        raise ValueError(
            f"image size {(image_height, image_width)} not divisible by {_PATCH_SIZE}"
        )
    backbone = load_dinov2_vitb14_reg_backbone(
        pretrained=pretrained, weights_path=weights_path
    )
    feature_dim = int(getattr(backbone, "embed_dim", _FEATURE_DIM))
    head = classification_head(feature_dim, num_attributes, dropout)
    return Dinov2AttributeClassifier(
        backbone,
        head,
        feature_dim=feature_dim,
        image_size=(image_height, image_width),
    )
