"""Tests for MixStyle (E12 / H12) and domain-balanced train batches."""

from __future__ import annotations

from pathlib import Path

import torch

from src.wacv.config import MixStyleConfig, load_config
from src.wacv.data import DomainBalancedBatchSampler
from src.wacv.models.convnext import build_convnext
from src.wacv.models.mixstyle import MixStyle


def test_mixstyle_is_identity_in_eval() -> None:
    module = MixStyle(p=1.0, alpha=0.1)
    module.eval()
    features = torch.randn(4, 8, 16, 8)
    assert torch.equal(module(features), features)


def test_mixstyle_changes_features_in_train() -> None:
    module = MixStyle(p=1.0, alpha=0.1)
    module.train()
    torch.manual_seed(0)
    features = torch.randn(8, 4, 8, 4)
    mixed = module(features)
    assert mixed.shape == features.shape
    assert not torch.allclose(mixed, features)


def test_convnext_injects_mixstyle_after_stages_1_and_2() -> None:
    config = MixStyleConfig(enabled=True, stages=[1, 2], p=0.5, alpha=0.1)
    model = build_convnext(
        "convnext_tiny",
        num_attributes=40,
        pretrained=False,
        dropout=0.0,
        mixstyle=config,
    )
    mixstyle_layers = [m for m in model.modules() if isinstance(m, MixStyle)]
    assert len(mixstyle_layers) == 2
    # features: stem, s1, MixStyle, down, s2, MixStyle, down, s3, down, s4
    assert isinstance(model.features[2], MixStyle)
    assert isinstance(model.features[5], MixStyle)
    model.eval()
    images = torch.randn(2, 3, 256, 128)
    logits = model(images)
    assert logits.shape == (2, 40)


def test_domain_balanced_batch_sampler_equal_counts() -> None:
    domains = ["PA100k"] * 40 + ["PETA"] * 24
    sampler = DomainBalancedBatchSampler(domains, batch_size=8, seed=7)
    assert sampler.per_domain == 4
    assert sampler.domain_order == ("PA100k", "PETA")
    for batch in sampler:
        assert len(batch) == 8
        labels = [domains[index] for index in batch]
        assert labels.count("PA100k") == 4
        assert labels.count("PETA") == 4


def test_e12_config_loads_mixstyle_and_domain_balanced() -> None:
    config = load_config(Path("configs/e12_mixstyle.yml"))
    assert config.run.name == "e12_mixstyle"
    assert config.model.mixstyle.enabled is True
    assert config.model.mixstyle.stages == [1, 2]
    assert config.model.mixstyle.p == 0.5
    assert config.model.mixstyle.alpha == 0.1
    assert config.data.domain_balanced is True
    assert config.data.train_domains == ["PA100k", "PETA"]
    assert config.ranking.distance == "hybrid"
    assert config.ranking.alpha == 0.5
    assert config.ranking.query_norm == "dual_softmax"
    assert config.ranking.query_norm_tau == 0.05
    assert config.inference.tta == ["identity", "hflip", "scale_crop"]
