"""Attribute prediction and Track 2 retrieval evaluation helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


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


def l1_distance(queries: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Compute `(queries, gallery)` L1 distances without a 3-D tensor."""
    query_values = queries.astype(np.float32)
    predicted_values = probabilities.astype(np.float32)
    return (
        query_values.sum(axis=1, keepdims=True)
        + predicted_values.sum(axis=1)[None, :]
        - 2 * query_values @ predicted_values.T
    )


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
