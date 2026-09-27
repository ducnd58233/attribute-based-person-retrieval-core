"""Attribute prediction and Track 2 retrieval evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping, Sequence
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.wacv.attributes import (
    ExclusiveAttributeGroup,
    exclusive_attribute_groups,
    independent_attribute_indices,
)
from src.wacv.query_adapter import AttributeModelWithHardQuery

DistanceName = Literal[
    "l1",
    "specker",
    "clear_hard",
    "nll",
    "hybrid",
    "group_categorical",
]
CalibrationMode = Literal["none", "per_attribute"]
TTAView = Literal["identity", "hflip", "scale_crop"]
_PROB_EPS = 1e-4
_DEFAULT_TTA: tuple[TTAView, ...] = ("identity",)

Scorer = Callable[..., np.ndarray]
SCORERS: MutableMapping[str, Scorer] = {}


def register_scorer(name: str, scorer: Scorer) -> None:
    """Register or replace a Track-2 ranking distance under ``name``."""
    SCORERS[name] = scorer


def get_scorer(name: str) -> Scorer:
    """Return a registered scorer or raise ``KeyError`` with known names."""
    try:
        return SCORERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(SCORERS)) or "(none)"
        raise KeyError(f"Unknown ranking distance {name!r}; known: {known}") from exc


def _unwrap_eval_module(model: nn.Module) -> nn.Module:
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
    probabilities: np.ndarray,
    temperature: float | np.ndarray,
) -> np.ndarray:
    """Temperature-scale clipped probabilities via logit space.

    ``temperature`` may be a scalar or a length-A vector (one scale per
    attribute). Values divide the logits (T>1 softens, T<1 sharpens).
    """
    clipped = np.clip(probabilities.astype(np.float32), _PROB_EPS, 1.0 - _PROB_EPS)
    temps = np.asarray(temperature, dtype=np.float32)
    if temps.ndim == 0:
        if float(temps) == 1.0:
            return clipped
        logits = np.log(clipped) - np.log1p(-clipped)
        scaled = logits / float(temps)
        return 1.0 / (1.0 + np.exp(-scaled))
    if temps.shape != (clipped.shape[1],):
        raise ValueError(
            f"temperature vector shape {temps.shape} != ({clipped.shape[1]},)"
        )
    if np.allclose(temps, 1.0):
        return clipped
    logits = np.log(clipped) - np.log1p(-clipped)
    scaled = logits / np.maximum(temps[None, :], 1e-3)
    return 1.0 / (1.0 + np.exp(-scaled))


def fit_per_attribute_temperatures(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    init: float = 1.0,
    max_iter: int = 50,
) -> np.ndarray:
    """Fit one temperature per attribute by minimising Bernoulli NLL (LBFGS).

    ``probabilities`` and ``targets`` are gallery rows ``(N, A)``. Fitting uses
    ID-val (or in-domain) labels only — never the hidden test set.
    """
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must share shape")
    clipped = np.clip(probabilities.astype(np.float64), _PROB_EPS, 1.0 - _PROB_EPS)
    labels = targets.astype(np.float64)
    num_attributes = clipped.shape[1]
    temperatures = np.full(num_attributes, float(init), dtype=np.float64)

    def _nll_one(temp: float, attr: int) -> float:
        scale = max(float(temp), 1e-3)
        logits = np.log(clipped[:, attr]) - np.log1p(-clipped[:, attr])
        calibrated = 1.0 / (1.0 + np.exp(-(logits / scale)))
        calibrated = np.clip(calibrated, _PROB_EPS, 1.0 - _PROB_EPS)
        y = labels[:, attr]
        return float(
            -np.mean(y * np.log(calibrated) + (1.0 - y) * np.log(1.0 - calibrated))
        )

    for attr in range(num_attributes):
        best_t = float(init)
        best_loss = _nll_one(best_t, attr)
        # Coarse grid then local refine (avoids scipy dependency)
        for candidate in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
            loss = _nll_one(candidate, attr)
            if loss < best_loss:
                best_loss = loss
                best_t = candidate
        # Coordinate descent around the best grid point
        step = 0.25
        for _ in range(max_iter):
            improved = False
            for delta in (-step, step):
                trial = max(best_t + delta, 1e-3)
                loss = _nll_one(trial, attr)
                if loss < best_loss - 1e-9:
                    best_loss = loss
                    best_t = trial
                    improved = True
            if not improved:
                step *= 0.5
                if step < 1e-3:
                    break
        temperatures[attr] = best_t
    return temperatures.astype(np.float32)


def apply_per_attribute_temperatures(
    probabilities: np.ndarray, temperatures: np.ndarray
) -> np.ndarray:
    """Apply a fitted per-attribute temperature vector to gallery probabilities."""
    return _calibrate_probabilities(probabilities, temperatures)


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


def nll_distance(
    queries: np.ndarray,
    probabilities: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Bernoulli negative log-likelihood distance (exact-match under independence).

    For query bit ``q`` and gallery probability ``p`` (optionally temperature-
    calibrated), contributes ``-(q log p + (1-q) log(1-p))``. Computed with two
    matmuls / broadcasts — never materialises a ``(Q, G, 40)`` tensor.
    """
    calibrated = _calibrate_probabilities(probabilities, temperature)
    log_p = np.log(calibrated)
    log_1mp = np.log(1.0 - calibrated)
    logits = log_p - log_1mp
    query_values = queries.astype(np.float32)
    # -q·logit(p) - 1·log(1-p)  ==  -Σ[q log p + (1-q) log(1-p)]
    return -(query_values @ logits.T) - log_1mp.sum(axis=1)[None, :]


def _bernoulli_nll_bits(
    queries: np.ndarray, calibrated: np.ndarray, indices: tuple[int, ...]
) -> np.ndarray:
    """Bernoulli NLL restricted to ``indices`` (already calibrated probs)."""
    if not indices:
        return np.zeros((queries.shape[0], calibrated.shape[0]), dtype=np.float32)
    idx = list(indices)
    query_bits = queries[:, idx].astype(np.float32)
    probs = calibrated[:, idx]
    log_p = np.log(probs)
    log_1mp = np.log(1.0 - probs)
    logits = log_p - log_1mp
    return (-(query_bits @ logits.T) - log_1mp.sum(axis=1)[None, :]).astype(np.float32)


def _group_categorical_term(
    queries: np.ndarray,
    calibrated: np.ndarray,
    group: ExclusiveAttributeGroup,
) -> np.ndarray:
    """Categorical NLL for one exclusive group; multi-hot falls back to Bernoulli."""
    idx = list(group.indices)
    query_bits = queries[:, idx].astype(np.float32)
    group_probs = calibrated[:, idx]
    if group.has_none:
        none_mass = np.clip(
            1.0 - group_probs.sum(axis=1, keepdims=True), _PROB_EPS, 1.0
        )
        extended_probs = np.concatenate([group_probs, none_mass], axis=1)
        none_query = (query_bits.sum(axis=1, keepdims=True) == 0).astype(np.float32)
        extended_queries = np.concatenate([query_bits, none_query], axis=1)
    else:
        extended_probs = group_probs
        extended_queries = query_bits

    row_sum = np.maximum(extended_probs.sum(axis=1, keepdims=True), _PROB_EPS)
    renorm = np.clip(extended_probs / row_sum, _PROB_EPS, 1.0)
    log_renorm = np.log(renorm)
    categorical = -(extended_queries @ log_renorm.T)
    bernoulli = _bernoulli_nll_bits(queries, calibrated, group.indices)

    ones = extended_queries.sum(axis=1) == 1.0
    multi = extended_queries.sum(axis=1) > 1.0
    zero = extended_queries.sum(axis=1) == 0.0
    use_categorical = ones
    use_bernoulli = multi | (zero & (not group.has_none))
    return (
        np.where(use_categorical[:, None], categorical, 0.0)
        + np.where(use_bernoulli[:, None], bernoulli, 0.0)
    ).astype(np.float32)


def group_categorical_nll_distance(
    queries: np.ndarray,
    probabilities: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Group-categorical NLL plus Bernoulli NLL on independent attributes.

    Exclusive groups (Age, Hair+none, Upper/Lower colour, Lower type) are
    renormalised to a simplex and scored with categorical log-likelihood when
    the query selects exactly one category. Multi-hot group queries fall back
    to Bernoulli NLL on that group's bits.
    """
    calibrated = _calibrate_probabilities(probabilities, temperature)
    num_attributes = calibrated.shape[1]
    distance = np.zeros((queries.shape[0], calibrated.shape[0]), dtype=np.float32)
    for group in exclusive_attribute_groups():
        distance += _group_categorical_term(queries, calibrated, group)
    independent = independent_attribute_indices(num_attributes)
    distance += _bernoulli_nll_bits(queries, calibrated, independent)
    return distance


def _minmax_per_query(distances: np.ndarray) -> np.ndarray:
    """Scale each query row to [0, 1]; constant rows become zeros."""
    row_min = distances.min(axis=1, keepdims=True)
    row_max = distances.max(axis=1, keepdims=True)
    spread = row_max - row_min
    safe = np.maximum(spread, 1e-8)
    normalised = (distances - row_min) / safe
    return np.where(spread > 0, normalised, np.zeros_like(distances))


def hybrid_distance(
    queries: np.ndarray,
    probabilities: np.ndarray,
    *,
    alpha: float = 0.5,
    temperature: float = 1.0,
) -> np.ndarray:
    """Per-query mix of min-max-normalised NLL and L1 distances."""
    nll = _minmax_per_query(
        nll_distance(queries, probabilities, temperature=temperature)
    )
    l1 = _minmax_per_query(l1_distance(queries, probabilities))
    weight = float(alpha)
    return weight * nll + (1.0 - weight) * l1


def ranking_distance(
    queries: np.ndarray,
    probabilities: np.ndarray,
    *,
    distance: DistanceName | str = "l1",
    temperature: float = 1.5,
    alpha: float = 0.5,
    gallery_embeddings: np.ndarray | None = None,
    query_embeddings: np.ndarray | None = None,
    clear_fusion: float = 0.5,
) -> np.ndarray:
    """Dispatch the configured Track-2 ranking distance via the scorer registry."""
    try:
        scorer = get_scorer(distance)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    return scorer(
        queries,
        probabilities,
        temperature=temperature,
        alpha=alpha,
        gallery_embeddings=gallery_embeddings,
        query_embeddings=query_embeddings,
        clear_fusion=clear_fusion,
    )


def _score_l1(
    queries: np.ndarray,
    probabilities: np.ndarray,
    **_: object,
) -> np.ndarray:
    return l1_distance(queries, probabilities)


def _score_specker(
    queries: np.ndarray,
    probabilities: np.ndarray,
    *,
    temperature: float = 1.5,
    **_: object,
) -> np.ndarray:
    return specker_distance(queries, probabilities, temperature=temperature)


def _score_clear_hard(
    queries: np.ndarray,
    probabilities: np.ndarray,
    *,
    temperature: float = 1.5,
    gallery_embeddings: np.ndarray | None = None,
    query_embeddings: np.ndarray | None = None,
    clear_fusion: float = 0.5,
    **_: object,
) -> np.ndarray:
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


def _score_nll(
    queries: np.ndarray,
    probabilities: np.ndarray,
    *,
    temperature: float = 1.0,
    **_: object,
) -> np.ndarray:
    return nll_distance(queries, probabilities, temperature=temperature)


def _score_hybrid(
    queries: np.ndarray,
    probabilities: np.ndarray,
    *,
    temperature: float = 1.0,
    alpha: float = 0.5,
    **_: object,
) -> np.ndarray:
    return hybrid_distance(queries, probabilities, alpha=alpha, temperature=temperature)


def _score_group_categorical(
    queries: np.ndarray,
    probabilities: np.ndarray,
    *,
    temperature: float = 1.0,
    **_: object,
) -> np.ndarray:
    return group_categorical_nll_distance(
        queries, probabilities, temperature=temperature
    )


register_scorer("l1", _score_l1)
register_scorer("specker", _score_specker)
register_scorer("clear_hard", _score_clear_hard)
register_scorer("nll", _score_nll)
register_scorer("hybrid", _score_hybrid)
register_scorer("group_categorical", _score_group_categorical)


def dual_softmax_rerank(
    distances: np.ndarray,
    *,
    tau: float = 0.05,
    chunk_cols: int = 4096,
) -> np.ndarray:
    """Query-set dual softmax over gallery columns (no labels, no updates).

    Converts distances ``D`` to similarities ``S = -D``, then
    ``S' = S + log_softmax_{queries}(S / tau)`` and returns ``-S'`` as a
    distance. Softmax is column-wise (over queries for each gallery image),
    which dampens hubness without using ground-truth matches.

    Gallery columns are processed in chunks so peak RAM stays bounded on
    Codabench-sized Q×G matrices (float64 full-matrix form peaked ~6GB).
    """
    similarities = -np.asarray(distances, dtype=np.float32)
    scale = max(float(tau), 1e-8)
    out = np.empty_like(similarities)
    n_gallery = similarities.shape[1]
    step = max(int(chunk_cols), 1)
    for start in range(0, n_gallery, step):
        end = min(start + step, n_gallery)
        block = similarities[:, start:end] / scale
        col_max = block.max(axis=0, keepdims=True)
        exp_shifted = np.exp(block - col_max)
        log_partition = np.log(
            np.maximum(exp_shifted.sum(axis=0, keepdims=True), 1e-12)
        )
        log_softmax = block - col_max - log_partition
        out[:, start:end] = -(similarities[:, start:end] + log_softmax)
    return out


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
