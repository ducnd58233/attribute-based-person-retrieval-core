"""Tests that data.image_size drives train/eval transforms (E11)."""

from __future__ import annotations

import torch
from PIL import Image

from src.wacv.config import load_config
from src.wacv.data import transforms_for_model


def test_default_transform_is_256x128() -> None:
    image = Image.new("RGB", (64, 128), color=(10, 20, 30))
    tensor = transforms_for_model("convnext_base", train=False)(image)
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (3, 256, 128)


def test_custom_image_size_384x192() -> None:
    image = Image.new("RGB", (64, 128), color=(10, 20, 30))
    tensor = transforms_for_model("convnext_base", train=False, image_size=(384, 192))(
        image
    )
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (3, 384, 192)


def test_e11_config_loads_image_size() -> None:
    from pathlib import Path

    config = load_config(Path("configs/e11_res_384.yml"))
    assert config.data.image_size == (384, 192)
    assert config.run.name == "e11_res_384"
