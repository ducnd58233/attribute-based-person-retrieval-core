"""Tests for flip + scale-crop test-time augmentation (E08)."""

from __future__ import annotations

import torch

from src.wacv.evaluation import apply_tta_view, predict_probabilities


def test_apply_tta_identity_and_hflip_shapes() -> None:
    images = torch.randn(2, 3, 256, 128)
    identity = apply_tta_view(images, "identity")
    flipped = apply_tta_view(images, "hflip")
    assert identity.shape == images.shape
    assert flipped.shape == images.shape
    assert torch.allclose(identity, images)
    assert torch.allclose(flipped, torch.flip(images, dims=(-1,)))


def test_apply_tta_scale_crop_returns_original_spatial_size() -> None:
    images = torch.randn(2, 3, 256, 128)
    cropped = apply_tta_view(images, "scale_crop", scale=1.15)
    assert cropped.shape == images.shape


def test_predict_probabilities_averages_tta_views() -> None:
    class _ConstModel(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # Distinct logits depending on whether the batch was flipped.
            flipped = images[..., 0].mean() < images[..., -1].mean()
            return torch.full(
                (images.shape[0], 4), 0.0 if flipped else 2.0, device=images.device
            )

    model = _ConstModel()
    images = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    loader = [(images, torch.zeros(2, 4))]
    single = predict_probabilities(
        model, loader, torch.device("cpu"), progress=False, tta=["identity"]
    )
    averaged = predict_probabilities(
        model,
        loader,
        torch.device("cpu"),
        progress=False,
        tta=["identity", "hflip"],
    )
    assert single.shape == (2, 4)
    assert averaged.shape == (2, 4)
    assert float(averaged.mean()) != float(single.mean())
