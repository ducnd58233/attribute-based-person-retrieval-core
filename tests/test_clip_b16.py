"""Tests for CLIP ViT-B/16 positional interpolation and transforms (E17)."""

from __future__ import annotations

import torch

from src.wacv.data import transforms_for_model
from src.wacv.models.clip import interpolate_clip_positional_embedding


def test_interpolate_pos_embed_to_16x8_grid() -> None:
    # Square 14x14 + CLS (ViT-B/16 @ 224)
    embedding = torch.randn(1 + 14 * 14, 768)
    out = interpolate_clip_positional_embedding(
        embedding, height=256, width=128, patch_size=16
    )
    assert out.shape == (1 + 16 * 8, 768)
    assert torch.allclose(out[0], embedding[0])


def test_clip_transforms_are_256x128() -> None:
    from PIL import Image

    image = Image.new("RGB", (64, 128), color=(10, 20, 30))
    tensor = transforms_for_model("clip_vit_b16", train=False)(image)
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (3, 256, 128)
