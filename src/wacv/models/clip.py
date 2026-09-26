"""CLIP ViT-B/16 vision tower as an attribute classifier (256x128 crops)."""

from __future__ import annotations

import math

import open_clip
import torch
import torch.nn.functional as F
from torch import nn

from src.wacv.models._common import classification_head

_CLIP_PATCH = 16
_TARGET_H = 256
_TARGET_W = 128


def interpolate_clip_positional_embedding(
    positional_embedding: torch.Tensor,
    *,
    height: int = _TARGET_H,
    width: int = _TARGET_W,
    patch_size: int = _CLIP_PATCH,
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
        """Freeze the vision tower (linear-probe stage)."""
        for parameter in self.visual.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the vision tower (full fine-tune stage)."""
        for parameter in self.visual.parameters():
            parameter.requires_grad = True

    def layerwise_param_groups(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
        layer_decay: float,
    ) -> list[dict[str, object]]:
        """AdamW param groups with deeper ViT blocks keeping higher LR."""
        blocks = list(self.visual.transformer.resblocks)
        num_layers = len(blocks)
        layer_id_by_param: dict[int, int] = {}
        for index, block in enumerate(blocks):
            for parameter in block.parameters():
                layer_id_by_param[id(parameter)] = index
        groups: list[dict[str, object]] = []
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        if head_params:
            groups.append(
                {
                    "params": head_params,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                }
            )
        early_params: list[nn.Parameter] = []
        for _name, parameter in self.visual.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in layer_id_by_param:
                continue
            early_params.append(parameter)
        if early_params:
            scale = layer_decay**num_layers
            groups.append(
                {
                    "params": early_params,
                    "lr": learning_rate * scale,
                    "weight_decay": weight_decay,
                }
            )
        for index, block in enumerate(blocks):
            params = [p for p in block.parameters() if p.requires_grad]
            if not params:
                continue
            scale = layer_decay ** (num_layers - 1 - index)
            groups.append(
                {
                    "params": params,
                    "lr": learning_rate * scale,
                    "weight_decay": weight_decay,
                }
            )
        return groups


def build_clip_attribute_classifier(
    *,
    num_attributes: int,
    pretrained: bool,
    dropout: float,
    arch: str = "ViT-B-16",
    pretrained_tag: str = "openai",
    image_height: int = _TARGET_H,
    image_width: int = _TARGET_W,
) -> ClipAttributeClassifier:
    """Build CLIP ViT-B/16 with pos embeddings for height x width crops."""
    model, _, _ = open_clip.create_model_and_transforms(
        arch,
        pretrained=pretrained_tag if pretrained else None,
    )
    visual = model.visual
    if not hasattr(visual, "positional_embedding"):
        raise TypeError(f"{arch} visual has no positional_embedding")
    resized = interpolate_clip_positional_embedding(
        visual.positional_embedding.data,
        height=image_height,
        width=image_width,
    )
    visual.positional_embedding = nn.Parameter(resized)
    visual.image_size = (image_height, image_width)
    with torch.inference_mode():
        probe = torch.zeros(1, 3, image_height, image_width)
        feature_dim = int(visual(probe).shape[-1])
    head = classification_head(feature_dim, num_attributes, dropout)
    return ClipAttributeClassifier(
        visual,
        head,
        feature_dim=feature_dim,
        image_size=(image_height, image_width),
    )
