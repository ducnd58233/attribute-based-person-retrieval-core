"""Attribute prediction and Track 2 retrieval evaluation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.wacv.query_adapter import AttributeModelWithHardQuery

DistanceName = Literal["l1", "specker", "clear_hard"]
_PROB_EPS = 1e-4


def _unwrap_eval_module(model: nn.Module) -> nn.Module:
    """Prefer EMA ``module`` attribute when present."""
    inner = getattr(model, "module", model)
    return inner if isinstance(inner, nn.Module) else model


def predict_probabilities(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    progress: bool,
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc="predict", disable=not progress):
            logits = model(images.to(device, non_blocking=True))
            batches.append(torch.sigmoid(logits).cpu().numpy())
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
    root = _unwrap_eval_module(model)
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


def l1_distance(queries: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Compute `(queries, gallery)` L1 distances without a 3-D tensor."""
    query_values = queries.astype(np.float32)
    predicted_values = probabilities.astype(np.float32)
    return (
        query_values.sum(axis=1, keepdims=True)
        + predicted_values.sum(axis=1)[None, :]
        - 2 * query_values @ predicted_values.T
    )


def _calibrate_probabilities(
    probabilities: np.ndarray, temperature: float
) -> np.ndarray:
    """Temperature-scale clipped probabilities via logit space."""
    clipped = np.clip(probabilities.astype(np.float32), _PROB_EPS, 1.0 - _PROB_EPS)
    if temperature == 1.0:
        return clipped
    logits = np.log(clipped) - np.log1p(-clipped)
    scaled = logits / float(temperature)
    return 1.0 / (1.0 + np.exp(-scaled))


def _attribute_reliability_weights(probabilities: np.ndarray) -> np.ndarray:
    """Higher weight for peaked, rare-leaning attributes (Specker-style)."""
    mean = probabilities.mean(axis=0)
    std = np.maximum(probabilities.std(axis=0), 1e-3)
    rarity = 1.0 / np.maximum(mean * (1.0 - mean), 1e-3)
    reliability = rarity / std
    return (reliability / np.maximum(reliability.mean(), 1e-8)).astype(np.float32)


def specker_distance(
    queries: np.ndarray,
    probabilities: np.ndarray,
    *,
    temperature: float = 1.5,
    attribute_weights: np.ndarray | None = None,
    use_reliability_weights: bool = False,
) -> np.ndarray:
    """Calibrated attribute distance (Specker ICIP 2021 ADAPT).

    Default: temperature-calibrate gallery probabilities, then L1 to binary
    queries. Optional reliability weights are off by default — on this
    ConvNeXt baseline they *hurt* val mADM in a controlled sweep.
    """
    calibrated = _calibrate_probabilities(probabilities, temperature)
    if attribute_weights is not None:
        weights = attribute_weights.astype(np.float32).ravel()
    elif use_reliability_weights:
        weights = _attribute_reliability_weights(calibrated)
    else:
        weights = np.ones(calibrated.shape[1], dtype=np.float32)
    if weights.shape[0] != calibrated.shape[1]:
        raise ValueError(
            f"attribute_weights length {weights.shape[0]} != "
            f"{calibrated.shape[1]} attributes"
        )
    query_values = queries.astype(np.float32)
    weighted_queries = query_values * weights[None, :]
    return (
        weighted_queries.sum(axis=1, keepdims=True)
        + (calibrated * weights[None, :]).sum(axis=1)[None, :]
        - 2.0 * weighted_queries @ calibrated.T
    )


def clear_hard_distance(
    queries: np.ndarray,
    probabilities: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    temperature: float = 1.5,
    clear_fusion: float = 0.5,
) -> np.ndarray:
    """Fuse Specker probability distance with CLEAR cosine (as a distance)."""
    specker = specker_distance(queries, probabilities, temperature=temperature)
    # embeddings are L2-normalized → cosine similarity in [-1, 1]
    cosine = (
        query_embeddings.astype(np.float32) @ gallery_embeddings.astype(np.float32).T
    )
    cosine_distance = 1.0 - cosine
    fusion = float(clear_fusion)
    return (1.0 - fusion) * specker + fusion * cosine_distance


def ranking_distance(
    queries: np.ndarray,
    probabilities: np.ndarray,
    *,
    distance: DistanceName = "l1",
    temperature: float = 1.5,
    gallery_embeddings: np.ndarray | None = None,
    query_embeddings: np.ndarray | None = None,
    clear_fusion: float = 0.5,
) -> np.ndarray:
    """Dispatch the configured Track-2 ranking distance."""
    if distance == "l1":
        return l1_distance(queries, probabilities)
    if distance == "specker":
        return specker_distance(queries, probabilities, temperature=temperature)
    if distance == "clear_hard":
        if gallery_embeddings is None or query_embeddings is None:
            raise ValueError(
                "clear_hard ranking requires gallery_embeddings and query_embeddings"
            )
        return clear_hard_distance(
            queries,
            probabilities,
            gallery_embeddings,
            query_embeddings,
            temperature=temperature,
            clear_fusion=clear_fusion,
        )
    raise ValueError(f"Unknown ranking distance: {distance}")


def mean_accuracy(probabilities: np.ndarray, targets: np.ndarray) -> float:
    predictions = probabilities >= 0.5
    positives = targets == 1
    negatives = ~positives
    true_positive_rate = (predictions & positives).sum(axis=0) / np.maximum(
        positives.sum(axis=0), 1
    )
    true_negative_rate = ((~predictions) & negatives).sum(axis=0) / np.maximum(
        negatives.sum(axis=0), 1
    )
    return float((0.5 * (true_positive_rate + true_negative_rate)).mean())
