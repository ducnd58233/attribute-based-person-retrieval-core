"""Tests for DINOv2 ViT-B/14-reg factory wiring and 252x126 transforms (E20)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from src.wacv.config import ModelConfig, load_config
from src.wacv.data import transforms_for_model
from src.wacv.models.dinov2 import (
    Dinov2AttributeClassifier,
    build_dinov2_vitb14_reg,
    load_dinov2_backbone,
)
from src.wacv.models.factory import build_model


class _FakeBlock(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.weight.mean()


class _FakeDinov2Backbone(nn.Module):
    """Minimal stand-in for torch.hub dinov2_vitb14_reg (no download)."""

    def __init__(self, embed_dim: int = 32, num_blocks: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList([_FakeBlock(embed_dim) for _ in range(num_blocks)])
        self.proj = nn.Linear(3, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        return self.proj(pooled)


def test_dinov2_transforms_are_252x126() -> None:
    from PIL import Image

    image = Image.new("RGB", (64, 128), color=(10, 20, 30))
    tensor = transforms_for_model(
        "dinov2_vitb14_reg", train=False, image_size=(252, 126)
    )(image)
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (3, 252, 126)


def test_e20_config_loads() -> None:
    config = load_config(Path("configs/e20_dinov2_lpft.yml"))
    assert config.model.name == "dinov2_vitb14_reg"
    assert config.data.image_size == (252, 126)
    assert config.optimization.lp_ft is True
    assert config.optimization.lp_epochs == 2
    assert config.optimization.layer_decay == 0.65
    assert config.ranking.distance == "hybrid"
    assert config.ranking.query_norm == "dual_softmax"


def test_build_dinov2_with_mocked_hub() -> None:
    fake = _FakeDinov2Backbone()
    with patch("src.wacv.models.dinov2.torch.hub.load", return_value=fake) as hub:
        model = build_dinov2_vitb14_reg(
            num_attributes=40,
            pretrained=False,
            dropout=0.1,
        )
    hub.assert_called_once()
    assert isinstance(model, Dinov2AttributeClassifier)
    assert model.image_size == (252, 126)
    assert model.feature_dim == 32
    images = torch.zeros(2, 3, 252, 126)
    logits = model(images)
    assert logits.shape == (2, 40)


def test_factory_dinov2_vitb14_reg_mocked() -> None:
    fake = _FakeDinov2Backbone()
    with patch("src.wacv.models.dinov2.torch.hub.load", return_value=fake):
        model = build_model(
            ModelConfig(name="dinov2_vitb14_reg", pretrained=False, dropout=0.0),
            num_attributes=40,
        )
    assert isinstance(model, Dinov2AttributeClassifier)
    model.freeze_backbone()
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    model.unfreeze_backbone()
    assert any(parameter.requires_grad for parameter in model.backbone.parameters())
    groups = model.layerwise_param_groups(
        learning_rate=1e-5, weight_decay=0.01, layer_decay=0.65
    )
    assert groups


def test_load_backbone_reads_local_weights(tmp_path: Path) -> None:
    fake = _FakeDinov2Backbone()
    weights_path = tmp_path / "dinov2_stub.pth"
    torch.save(fake.state_dict(), weights_path)
    loaded = _FakeDinov2Backbone()
    with patch("src.wacv.models.dinov2.torch.hub.load", return_value=loaded) as hub:
        backbone = load_dinov2_backbone(pretrained=True, weights_path=weights_path)
    assert hub.call_args.kwargs.get("pretrained") is False
    assert backbone is loaded


def test_build_rejects_non_divisible_image_size() -> None:
    with patch(
        "src.wacv.models.dinov2.torch.hub.load",
        return_value=_FakeDinov2Backbone(),
    ):
        try:
            build_dinov2_vitb14_reg(
                num_attributes=40,
                pretrained=False,
                dropout=0.0,
                image_height=250,
                image_width=126,
            )
        except ValueError as exc:
            assert "not divisible" in str(exc)
        else:
            raise AssertionError("expected ValueError for non-divisible height")
