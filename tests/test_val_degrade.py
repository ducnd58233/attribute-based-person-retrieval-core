"""Unit tests for T28/E81 fixed V-deg transforms (organizer-val only)."""

from __future__ import annotations

from PIL import Image

from src.wacv.data import (
    VAL_DEGRADE_LEVELS,
    FixedDownscaleUpsample,
    FixedJpegCompression,
    val_degrade_transform,
)


def test_val_degrade_levels_are_l1_l2_l3() -> None:
    assert set(VAL_DEGRADE_LEVELS) == {"L1", "L2", "L3"}


def test_fixed_downscale_is_deterministic() -> None:
    image = Image.new("RGB", (64, 128), color=(40, 80, 120))
    op = FixedDownscaleUpsample(0.40)
    assert list(op(image).getdata()) == list(op(image).getdata())


def test_fixed_jpeg_is_deterministic() -> None:
    image = Image.new("RGB", (32, 64), color=(10, 20, 30))
    op = FixedJpegCompression(55)
    assert list(op(image).getdata()) == list(op(image).getdata())


def test_val_degrade_transform_returns_tensor_shape() -> None:
    image = Image.new("RGB", (100, 200), color=(5, 6, 7))
    pipeline = val_degrade_transform(
        (224, 224),
        "L2",
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    tensor = pipeline(image)
    assert tuple(tensor.shape) == (3, 224, 224)
