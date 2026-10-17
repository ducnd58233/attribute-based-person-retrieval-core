"""Latent kNN smoothing of gallery attribute probabilities (E74).

Each gallery image's probabilities are blended with those of its nearest
gallery neighbours in the model's pooled-embedding space. The same algorithm
is mirrored in ``submission/run.py`` when ``model_card.json`` carries a
``neighbour_smooth`` block (packaging default for CLIP/DINOv2).
"""

from __future__ import annotations

import numpy as np

from src.wacv.config import NeighbourSmoothConfig

_CHUNK = 2048


def neighbour_smooth_probabilities(
    probabilities: np.ndarray,
    embeddings: np.ndarray,
    config: NeighbourSmoothConfig,
) -> np.ndarray:
    """Return ``(1 - beta) * p + beta * mean(kept neighbour p)`` for every image.

    Neighbours are the ``config.k`` most cosine-similar gallery images (self
    excluded); those below ``config.min_similarity`` are dropped, and an image
    left with none keeps its own probabilities.
    """
    if len(probabilities) != len(embeddings):
        raise ValueError(
            f"{len(probabilities)} probability rows but {len(embeddings)} embeddings"
        )
    count = len(probabilities)
    k = min(config.k, count - 1)
    if k < 1:
        return probabilities.copy()
    unit = embeddings.astype(np.float32)
    unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    smoothed = probabilities.copy()
    for start in range(0, count, _CHUNK):
        similarity = unit[start : start + _CHUNK] @ unit.T
        rows = np.arange(similarity.shape[0])
        similarity[rows, start + rows] = -np.inf
        nearest = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        kept = np.take_along_axis(similarity, nearest, axis=1) >= config.min_similarity
        weights = kept.astype(np.float32)
        kept_count = weights.sum(axis=1)
        has_neighbours = kept_count > 0
        neighbour_sum = (probabilities[nearest] * weights[..., None]).sum(axis=1)
        neighbour_mean = (
            neighbour_sum[has_neighbours] / kept_count[has_neighbours, None]
        )
        block = smoothed[start : start + similarity.shape[0]]
        block[has_neighbours] = (1.0 - config.beta) * block[
            has_neighbours
        ] + config.beta * neighbour_mean
    return smoothed
