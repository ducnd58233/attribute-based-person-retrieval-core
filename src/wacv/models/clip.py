"""CLIP vision towers as attribute classifiers (B/16 @ 256x128, L/14 @ 224x112)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import open_clip
import torch
import torch.nn.functional as F
from torch import nn

from src.wacv.models._common import (
    build_layerwise_param_groups,
    classification_head,
    set_requires_grad,
)
from src.wacv.models.lora import inject_lora_into_linear

_CLIP_PATCH_B16 = 16
_CLIP_PATCH_L14 = 14
_TARGET_H_B16 = 256
_TARGET_W_B16 = 128
_TARGET_H_L14 = 224
_TARGET_W_L14 = 112


@dataclass(frozen=True)
class ClipTowerSpec:
    """open_clip architecture + crop geometry for positional interpolation."""

    arch: str
    patch_size: int
    image_height: int
    image_width: int
    pretrained_tag: str = "openai"


_CLIP_B16 = ClipTowerSpec(
    arch="ViT-B-16",
    patch_size=_CLIP_PATCH_B16,
    image_height=_TARGET_H_B16,
    image_width=_TARGET_W_B16,
)
_CLIP_L14 = ClipTowerSpec(
    arch="ViT-L-14",
    patch_size=_CLIP_PATCH_L14,
    image_height=_TARGET_H_L14,
    image_width=_TARGET_W_L14,
)


def interpolate_clip_positional_embedding(
    positional_embedding: torch.Tensor,
    *,
    height: int = _TARGET_H_B16,
    width: int = _TARGET_W_B16,
    patch_size: int = _CLIP_PATCH_B16,
) -> torch.Tensor:
    """Resize CLIP patch positional embeddings to a non-square grid.

    ``positional_embedding`` is ``[1 + grid*grid, dim]`` (CLS + square patches).
    Returns ``[1 + (H/P)*(W/P), dim]`` with CLS unchanged.
    """
    if positional_embedding.ndim != 2:
        raise ValueError(
            "expected [tokens, dim] positional embedding, "
            f"got {tuple(positional_embedding.shape)}"
        )
    _, dim = positional_embedding.shape
    cls_token = positional_embedding[:1]
    patch_tokens = positional_embedding[1:]
    src_grid = int(math.sqrt(patch_tokens.shape[0]))
    if src_grid * src_grid != patch_tokens.shape[0]:
        raise ValueError(
            f"non-square source grid from {patch_tokens.shape[0]} patch tokens"
        )
    dst_h = height // patch_size
    dst_w = width // patch_size
    if dst_h * patch_size != height or dst_w * patch_size != width:
        raise ValueError(f"image size {(height, width)} not divisible by {patch_size}")
    if dst_h == src_grid and dst_w == src_grid:
        return positional_embedding
    reshaped = patch_tokens.reshape(1, src_grid, src_grid, dim).permute(0, 3, 1, 2)
    resized = F.interpolate(
        reshaped.float(),
        size=(dst_h, dst_w),
        mode="bicubic",
        align_corners=False,
    )
    flat = resized.permute(0, 2, 3, 1).reshape(dst_h * dst_w, dim)
    flat = flat.to(dtype=positional_embedding.dtype)
    return torch.cat([cls_token, flat], dim=0)


class ClipAttributeClassifier(nn.Module):
    """open_clip visual encoder + linear attribute head."""

    def __init__(
        self,
        visual: nn.Module,
        head: nn.Module,
        *,
        feature_dim: int,
        image_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.visual = visual
        self.head = head
        self.feature_dim = feature_dim
        self.image_size = image_size
        # Lets HardQueryAdapter infer dim the same way as ConvNeXt.
        self.classifier = head

    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        """CLIP image features (open_clip encode_image path)."""
        return self.visual(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Attribute logits."""
        return self.head(self.encode_features(images))

    def freeze_backbone(self) -> None:
        """Freeze the vision tower; keep LoRA adapter weights trainable."""
        for name, parameter in self.visual.named_parameters():
            if "lora_" in name:
                continue
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the vision tower (full fine-tune stage)."""
        set_requires_grad(self.visual, True)

    def inject_attention_lora(self, *, rank: int, alpha: float = 16.0) -> int:
        """Parametrize each attention ``out_proj.weight`` with LoRA; return count.

        Base visual weights stay frozen; only LoRA A/B (and the attribute head)
        remain trainable. Uses attention out-projections only (fused in_proj is
        left frozen) so packaging stays peft-free. Parametrization is required
        because ``MultiheadAttention`` applies ``out_proj`` via ``F.linear`` on
        ``.weight``, not ``out_proj.forward``.
        """
        self.freeze_backbone()
        replaced = 0
        for block in self.visual.transformer.resblocks:
            inject_lora_into_linear(block.attn.out_proj, rank=rank, alpha=alpha)
            replaced += 1
        return replaced

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
            tower=self.visual,
            blocks=self.visual.transformer.resblocks,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            layer_decay=layer_decay,
        )


def build_clip_attribute_classifier(
    *,
    num_attributes: int,
    pretrained: bool,
    dropout: float,
    tower: ClipTowerSpec = _CLIP_B16,
    lora_rank: int = 0,
    lora_alpha: float = 16.0,
) -> ClipAttributeClassifier:
    """Build a CLIP vision tower with pos embeddings for the tower crop size."""
    model, _, _ = open_clip.create_model_and_transforms(
        tower.arch,
        pretrained=tower.pretrained_tag if pretrained else None,
    )
    visual = model.visual
    if not hasattr(visual, "positional_embedding"):
        raise TypeError(f"{tower.arch} visual has no positional_embedding")
    resized = interpolate_clip_positional_embedding(
        visual.positional_embedding.data,
        height=tower.image_height,
        width=tower.image_width,
        patch_size=tower.patch_size,
    )
    visual.positional_embedding = nn.Parameter(resized)
    visual.image_size = (tower.image_height, tower.image_width)
    with torch.inference_mode():
        probe = torch.zeros(1, 3, tower.image_height, tower.image_width)
        feature_dim = int(visual(probe).shape[-1])
    head = classification_head(feature_dim, num_attributes, dropout)
    classifier = ClipAttributeClassifier(
        visual,
        head,
        feature_dim=feature_dim,
        image_size=(tower.image_height, tower.image_width),
    )
    if lora_rank > 0:
        classifier.inject_attention_lora(rank=lora_rank, alpha=lora_alpha)
    return classifier


def build_clip_vit_l14(
    *,
    num_attributes: int,
    pretrained: bool,
    dropout: float,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
) -> ClipAttributeClassifier:
    """CLIP ViT-L/14 at 224x112 with attention out_proj LoRA (PromptPAR-lite)."""
    return build_clip_attribute_classifier(
        num_attributes=num_attributes,
        pretrained=pretrained,
        dropout=dropout,
        tower=_CLIP_L14,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
