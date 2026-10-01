"""Deterministic V-deg (val degrade) transforms for T28."""

from __future__ import annotations

from PIL import Image

from src.wacv.data import (
    VAL_DEGRADE_LEVELS,
    color_jitter_hue_span,
    val_degrade_transform,
)


def test_val_degrade_levels_match_hypothesis() -> None:
    assert VAL_DEGRADE_LEVELS["L1"]["downscale"] == 0.60
    assert VAL_DEGRADE_LEVELS["L1"]["blur_sigma"] == 0.6
    assert VAL_DEGRADE_LEVELS["L1"]["jpeg_quality"] == 75
    assert VAL_DEGRADE_LEVELS["L2"]["downscale"] == 0.40
    assert VAL_DEGRADE_LEVELS["L3"]["downscale"] == 0.28


def test_val_degrade_is_deterministic() -> None:
    image = Image.new("RGB", (128, 256), color=(40, 80, 120))
    transform = val_degrade_transform(
        (256, 128),
        "L2",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    a = transform(image)
    b = transform(image)
    assert a.shape == b.shape
    assert (a - b).abs().max().item() == 0.0


def test_val_degrade_has_no_hue_jitter() -> None:
    transform = val_degrade_transform(
        (256, 128),
        "L1",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    assert color_jitter_hue_span(transform) is None
