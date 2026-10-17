"""CLIP vision towers as attribute classifiers (B/16 @ 256x128, L/14 @ 224x112)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast, get_args

import open_clip
import torch
import torch.nn.functional as F
from torch import nn

from src.wacv.config import MODEL_NATIVE_IMAGE_SIZE, LoraTarget
from src.wacv.models._common import (
    build_layerwise_param_groups,
    classification_head,
    set_requires_grad,
)
from src.wacv.models.lora import inject_lora_into_linear, inject_lora_into_parameter


class _SizedVisual(Protocol):
    image_size: tuple[int, int]


def _resblocks(visual: nn.Module) -> nn.ModuleList | nn.Sequential:
    """Transformer blocks of an open_clip vision tower."""
    blocks = visual.get_submodule("transformer.resblocks")
    if not isinstance(blocks, nn.ModuleList | nn.Sequential):
        raise TypeError(f"expected a sequence of blocks, got {type(blocks)}")
    return blocks


@dataclass(frozen=True)
class ClipTowerSpec:
    """open_clip architecture + crop geometry for positional interpolation."""

    arch: str
    patch_size: int
    image_height: int
    image_width: int
    pretrained_tag: str = "openai"


def _tower_spec(arch: str, patch_size: int, model_name: str) -> ClipTowerSpec:
    height, width = MODEL_NATIVE_IMAGE_SIZE[model_name]
    return ClipTowerSpec(
        arch=arch, patch_size=patch_size, image_height=height, image_width=width
    )


_CLIP_B16 = _tower_spec("ViT-B-16", 16, "clip_vit_b16")
_CLIP_L14 = _tower_spec("ViT-L-14", 14, "clip_vit_l14")


def interpolate_clip_positional_embedding(
    positional_embedding: torch.Tensor,
    *,
    height: int,
    width: int,
    patch_size: int,
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

    def add_lora_adapters(
        self,
        *,
        rank: int,
        alpha: float = 16.0,
        targets: tuple[LoraTarget, ...] = ("out_proj",),
    ) -> int:
        """Add LoRA to the chosen layers of every transformer block; return count.

        ``out_proj``: attention output projection. ``in_proj``: the fused q/k/v
        projection weight of ``MultiheadAttention``. ``mlp``: both MLP Linears.
        Base visual weights stay frozen; only LoRA A/B (and the attribute head)
        train. LoRA is a weight parametrization because ``MultiheadAttention``
        reads its weights directly instead of calling the Linear modules.
        """
        unknown = set(targets) - set(get_args(LoraTarget))
        if unknown:
            raise ValueError(f"unknown LoRA targets {sorted(unknown)}")
        self.freeze_backbone()
        adapted = 0
        for block in _resblocks(self.visual):
            linear_names = []
            if "out_proj" in targets:
                linear_names.append("attn.out_proj")
            if "mlp" in targets:
                linear_names += ["mlp.c_fc", "mlp.c_proj"]
            for name in linear_names:
                linear = block.get_submodule(name)
                if not isinstance(linear, nn.Linear):
                    raise TypeError(f"expected {name} to be Linear, got {type(linear)}")
                inject_lora_into_linear(linear, rank=rank, alpha=alpha)
                adapted += 1
            if "in_proj" in targets:
                inject_lora_into_parameter(
                    block.get_submodule("attn"),
                    "in_proj_weight",
                    rank=rank,
                    alpha=alpha,
                )
                adapted += 1
        return adapted

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
            blocks=_resblocks(self.visual),
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
    lora_targets: tuple[LoraTarget, ...] = ("out_proj",),
) -> ClipAttributeClassifier:
    """Build a CLIP vision tower with pos embeddings for the tower crop size."""
    model, _, _ = open_clip.create_model_and_transforms(
        tower.arch,
        pretrained=tower.pretrained_tag if pretrained else None,
    )
    visual = cast(nn.Module, model.visual)
    if not hasattr(visual, "positional_embedding"):
        raise TypeError(f"{tower.arch} visual has no positional_embedding")
    resized = interpolate_clip_positional_embedding(
        visual.get_parameter("positional_embedding").data,
        height=tower.image_height,
        width=tower.image_width,
        patch_size=tower.patch_size,
    )
    visual.register_parameter("positional_embedding", nn.Parameter(resized))
    cast(_SizedVisual, visual).image_size = (tower.image_height, tower.image_width)
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
        classifier.add_lora_adapters(
            rank=lora_rank, alpha=lora_alpha, targets=lora_targets
        )
    return classifier


def build_clip_vit_l14(
    *,
    num_attributes: int,
    pretrained: bool,
    dropout: float,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    lora_targets: tuple[LoraTarget, ...] = ("out_proj",),
) -> ClipAttributeClassifier:
    """CLIP ViT-L/14 at 224x112 with LoRA on ``lora_targets`` (PromptPAR-lite)."""
    return build_clip_attribute_classifier(
        num_attributes=num_attributes,
        pretrained=pretrained,
        dropout=dropout,
        tower=_CLIP_L14,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_targets=lora_targets,
    )
