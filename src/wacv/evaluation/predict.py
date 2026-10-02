"""Gallery attribute prediction: test-time augmentation views and embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.wacv.config import TTAView
from src.wacv.query_adapter import AttributeModelWithHardQuery

_DEFAULT_TTA: tuple[TTAView, ...] = ("identity",)


def unwrap_eval_module(model: nn.Module) -> nn.Module:
    """Prefer EMA ``module`` attribute when present."""
    inner = getattr(model, "module", model)
    return inner if isinstance(inner, nn.Module) else model


def apply_tta_view(
    images: torch.Tensor,
    view: str,
    *,
    scale: float = 1.15,
) -> torch.Tensor:
    """Return a tensor view for test-time augmentation (NCHW)."""
    if view == "identity":
        return images
    if view == "hflip":
        return torch.flip(images, dims=(-1,))
    if view == "scale_crop":
        _, _, height, width = images.shape
        factor = float(scale)
        if factor <= 0.0:
            raise ValueError(f"tta scale must be positive, got {factor}")
        new_h = max(height, round(height * factor))
        new_w = max(width, round(width * factor))
        upsampled = F.interpolate(
            images,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        top = max(0, (new_h - height) // 2)
        left = max(0, (new_w - width) // 2)
        return upsampled[:, :, top : top + height, left : left + width]
    raise ValueError(f"Unknown TTA view {view!r}")


def predict_probabilities(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    progress: bool,
    tta: Sequence[str] | None = None,
    tta_scale: float = 1.15,
) -> np.ndarray:
    """Predict gallery attribute probabilities, optionally averaging TTA views."""
    model.eval()
    views: Sequence[str] = tuple(tta) if tta else _DEFAULT_TTA
    if not views:
        views = _DEFAULT_TTA
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc="predict", disable=not progress):
            batch = images.to(device, non_blocking=True)
            probs_sum: torch.Tensor | None = None
            for view in views:
                logits = model(apply_tta_view(batch, view, scale=tta_scale))
                probs = torch.sigmoid(logits)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            assert probs_sum is not None
            batches.append((probs_sum / float(len(views))).cpu().numpy())
    if not batches:
        raise ValueError("Evaluation loader is empty")
    return np.concatenate(batches, axis=0)


def predict_gallery_embeddings(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    progress: bool,
) -> np.ndarray:
    """Encode gallery images with the CLEAR-lite hard-query adapter."""
    model.eval()
    root = unwrap_eval_module(model)
    if not isinstance(root, AttributeModelWithHardQuery):
        raise TypeError(
            "clear_hard ranking requires AttributeModelWithHardQuery; "
            f"got {type(root)}"
        )
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc="embed", disable=not progress):
            embeddings = root.gallery_embeddings(images.to(device, non_blocking=True))
            batches.append(embeddings.cpu().numpy())
    if not batches:
        raise ValueError("Evaluation loader is empty")
    return np.concatenate(batches, axis=0)
