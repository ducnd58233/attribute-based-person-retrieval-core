"""Smoke/unit tests for SOLIDER Swin-B (E21) with mocked weights."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from src.wacv.config import ModelConfig, load_config
from src.wacv.data import transforms_for_model
from src.wacv.models.factory import build_model
from src.wacv.models.solider import (
    SOLIDER_FEATURE_DIM,
    SOLIDER_SEMANTIC_WEIGHT_PAR,
    SOLIDER_SWIN_B_DRIVE_FILE_ID,
    SOLIDER_SWIN_B_EXPECTED_FILENAME,
    SoliderAttributeClassifier,
    build_solider_swin_b,
    resolve_solider_swin_b_weights,
    verify_weights_sha256,
)


def test_e21_config_loads_solider_swin_b() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = load_config(repo / "configs" / "e21_solider_swin_b.yml")
    assert config.model.name == "solider_swin_b"
    assert config.model.semantic_weight == SOLIDER_SEMANTIC_WEIGHT_PAR
    assert config.data.train_domains == ["PA100k", "PETA"]
    assert config.data.image_size == (256, 128)
    assert config.ranking.distance == "hybrid"
    assert config.ranking.query_norm == "dual_softmax"
    assert config.inference.tta == ["identity", "hflip", "scale_crop"]
    # Provenance knobs stay configurable without requiring a local file.
    assert config.model.weights_path is None
    assert config.model.pretrained_url is None
    assert SOLIDER_SWIN_B_DRIVE_FILE_ID == "1uh7tO34tMf73MJfFqyFEGx42UBktTbZU"
    assert SOLIDER_SWIN_B_EXPECTED_FILENAME == "solider_swin_base.pth"


def test_solider_transforms_are_imagenet_256x128() -> None:
    image = Image.new("RGB", (64, 128), color=(10, 20, 30))
    tensor = transforms_for_model("solider_swin_b", train=False)(image)
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (3, 256, 128)


def test_build_solider_without_pretrained_forward() -> None:
    config = ModelConfig(name="solider_swin_b", pretrained=False, dropout=0.0)
    model = build_solider_swin_b(config, num_attributes=40)
    assert isinstance(model, SoliderAttributeClassifier)
    assert model.feature_dim == SOLIDER_FEATURE_DIM
    assert model.semantic_weight == SOLIDER_SEMANTIC_WEIGHT_PAR
    images = torch.randn(2, 3, 256, 128)
    logits = model(images)
    assert logits.shape == (2, 40)
    features = model.encode_features(images)
    assert features.shape == (2, SOLIDER_FEATURE_DIM)


def test_factory_solider_with_mocked_checkpoint(tmp_path: Path) -> None:
    # Random init backbone, save its state_dict, reload via weights_path.
    seed = ModelConfig(name="solider_swin_b", pretrained=False, dropout=0.1)
    donor = build_solider_swin_b(seed, num_attributes=40)
    ckpt = tmp_path / SOLIDER_SWIN_B_EXPECTED_FILENAME
    torch.save(donor.backbone.state_dict(), ckpt)

    config = ModelConfig(
        name="solider_swin_b",
        pretrained=True,
        dropout=0.1,
        weights_path=ckpt,
        semantic_weight=0.8,
    )
    model = build_model(config, num_attributes=40)
    assert isinstance(model, SoliderAttributeClassifier)
    logits = model(torch.randn(1, 3, 256, 128))
    assert logits.shape == (1, 40)


def test_sha256_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "solider_swin_base.pth"
    path.write_bytes(b"not-a-real-checkpoint")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_weights_sha256(path, expected="0" * 64)


def test_resolve_requires_file_when_weights_path_set(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError, match="weights_path"):
        resolve_solider_swin_b_weights(
            weights_path=missing,
            pretrained=True,
            pretrained_url=None,
            weights_sha256=None,
        )


def test_e21_yaml_documents_provenance() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "configs" / "e21_solider_swin_b.yml").read_text(encoding="utf-8")
    assert "1uh7tO34tMf73MJfFqyFEGx42UBktTbZU" in text
    assert "solider_swin_base.pth" in text
    assert "pretrain-only" in text.lower() or "Pretrain-only" in text
    payload = yaml.safe_load(text)
    assert payload["model"]["name"] == "solider_swin_b"
