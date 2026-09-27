"""Tests for CLIP ViT-L/14 LoRA PEFT path (E19)."""

from __future__ import annotations

import torch
from PIL import Image

from src.wacv.data import transforms_for_model
from src.wacv.models.clip import interpolate_clip_positional_embedding
from src.wacv.models.lora import LoRALinear


def test_l14_pos_embed_224x112() -> None:
    # Square 16x16 + CLS (ViT-L/14 @ 224)
    embedding = torch.randn(1 + 16 * 16, 1024)
    out = interpolate_clip_positional_embedding(
        embedding, height=224, width=112, patch_size=14
    )
    assert out.shape == (1 + 16 * 8, 1024)


def test_clip_l14_transforms_are_224x112() -> None:
    image = Image.new("RGB", (64, 128), color=(10, 20, 30))
    tensor = transforms_for_model("clip_vit_l14", train=False)(image)
    assert tuple(tensor.shape) == (3, 224, 112)


def test_lora_linear_preserves_shape_and_starts_near_base() -> None:
    base = torch.nn.Linear(8, 4)
    wrapped = LoRALinear(base, rank=2, alpha=4.0)
    inputs = torch.randn(3, 8)
    with torch.inference_mode():
        assert torch.allclose(wrapped(inputs), base(inputs), atol=1e-6)
    assert wrapped.lora_A.requires_grad
    assert not base.weight.requires_grad
