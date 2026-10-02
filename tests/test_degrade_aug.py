"""Unit tests for surveillance degradation augmentation (E13)."""

from __future__ import annotations

from pathlib import Path

from torchvision import transforms

from src.wacv.config import load_config
from src.wacv.data import (
    RandomDownscaleUpsample,
    RandomJpegCompression,
    transforms_for_model,
)


def test_e13_config_loads_degrade_aug() -> None:
    config = load_config(Path("configs/e13_degrade_aug.yml"))
    assert config.run.name == "e13_degrade_aug"
    assert config.data.degrade_aug.enabled is True
    assert config.data.degrade_aug.downscale == (0.3, 1.0)
    assert config.data.degrade_aug.blur_p == 0.3
    assert config.data.degrade_aug.jpeg == (30, 90)
    assert config.data.train_domains == ["PA100k", "PETA"]
    assert config.ranking.distance == "hybrid"
    assert config.ranking.alpha == 0.5
    assert config.ranking.query_norm == "dual_softmax"
    assert config.ranking.query_norm_tau == 0.05
    assert config.inference.tta == ["identity", "hflip", "scale_crop"]


def test_degrade_aug_disables_hue_jitter() -> None:
    config = load_config(Path("configs/e13_degrade_aug.yml"))
    pipeline = transforms_for_model(
        config.model.name,
        train=True,
        image_size=config.data.image_size,
        degrade_aug=config.data.degrade_aug,
    )
    jitters = [
        step for step in pipeline.transforms if isinstance(step, transforms.ColorJitter)
    ]
    assert len(jitters) == 1
    assert jitters[0].hue is None


def test_degrade_aug_pipeline_includes_degradation_steps() -> None:
    config = load_config(Path("configs/e13_degrade_aug.yml"))
    pipeline = transforms_for_model(
        config.model.name,
        train=True,
        image_size=config.data.image_size,
        degrade_aug=config.data.degrade_aug,
    )
    step_types = {type(step) for step in pipeline.transforms}
    assert RandomDownscaleUpsample in step_types
    assert RandomJpegCompression in step_types
    blur_steps = [
        step
        for step in pipeline.transforms
        if isinstance(step, transforms.RandomApply)
        and any(isinstance(inner, transforms.GaussianBlur) for inner in step.transforms)
    ]
    assert len(blur_steps) == 1
    assert blur_steps[0].p == config.data.degrade_aug.blur_p
