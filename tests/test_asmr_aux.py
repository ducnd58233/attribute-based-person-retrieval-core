"""Unit tests for ASMR weighted Hamming and semantic-margin auxiliary loss."""

from __future__ import annotations

import torch

from src.wacv.losses import asmr_semantic_margin_loss, pairwise_weighted_hamming
from src.wacv.query_adapter import HardQueryAdapter


def test_pairwise_weighted_hamming_toy_batch() -> None:
    queries = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    weights = torch.tensor([1.0, 2.0, 1.0])
    dist = pairwise_weighted_hamming(queries, weights)
    expected = torch.tensor(
        [
            [0.0, 3.0, 2.0],
            [3.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    torch.testing.assert_close(dist, expected)


def test_asmr_semantic_margin_loss_penalizes_near_violation() -> None:
    gallery = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1], dtype=torch.long)
    margins = torch.tensor([[0.0, 0.1], [0.1, 0.0]])
    loss = asmr_semantic_margin_loss(gallery, queries, labels, margins)
    assert loss.item() > 0.0


def test_hard_query_adapter_semantic_margin_matrix() -> None:
    adapter = HardQueryAdapter(num_attributes=3, feature_dim=8, adapter_dim=16)
    bits = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    margins = adapter.semantic_margin_matrix(bits)
    assert margins.shape == (2, 2)
    assert margins[0, 0].item() == 0.0
    assert margins[1, 1].item() == 0.0
    assert margins[0, 1].item() > 0.0


def test_pooled_features_supports_encode_features_api() -> None:
    """CLIP-style backbones expose encode_features instead of ConvNeXt features."""
    from src.wacv.query_adapter import pooled_features

    class _FakeClip(torch.nn.Module):
        def encode_features(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(2, 3))

    images = torch.randn(2, 3, 8, 4)
    out = pooled_features(_FakeClip(), images)
    assert out.shape == (2, 3)
