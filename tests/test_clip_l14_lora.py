"""Tests for CLIP ViT-L/14 LoRA PEFT path (E19)."""

from __future__ import annotations

import torch
from PIL import Image
from torch.nn.utils.parametrize import is_parametrized

from src.wacv.data import transforms_for_model
from src.wacv.models.clip import interpolate_clip_positional_embedding
from src.wacv.models.lora import inject_lora_into_linear


def test_l14_pos_embed_224x112() -> None:
    # Square 16x16 + CLS (ViT-L/14 @ 224)
    embedding = torch.randn(1 + 16 * 16, 1024)
    out = interpolate_clip_positional_embedding(
        embedding, height=224, width=112, patch_size=14
    )
    assert out.shape == (1 + 16 * 8, 1024)


def test_clip_l14_transforms_are_224x112() -> None:
    image = Image.new("RGB", (64, 128), color=(10, 20, 30))
    tensor = transforms_for_model("clip_vit_l14", train=False, image_size=(224, 112))(
        image
    )
    assert tuple(tensor.shape) == (3, 224, 112)


def test_lora_parametrization_keeps_weight_attr_and_starts_as_base() -> None:
    base = torch.nn.Linear(8, 4)
    original = base.weight.detach().clone()
    inject_lora_into_linear(base, rank=2, alpha=4.0)
    assert is_parametrized(base, "weight")
    assert hasattr(base, "weight")
    assert torch.allclose(base.weight, original, atol=1e-6)
    # MHA-style F.linear path must still work
    inputs = torch.randn(3, 8)
    out = torch.nn.functional.linear(inputs, base.weight, base.bias)
    assert out.shape == (3, 4)
    lora_params = [
        p for n, p in base.named_parameters() if "lora_" in n and p.requires_grad
    ]
    assert lora_params
    assert not base.parametrizations.weight.original.requires_grad
