"""One predict -> calibrate -> rank -> score sequence for a gallery split."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.wacv.config import InferenceConfig, RankingConfig
from src.wacv.evaluation.calibration import (
    calibrate_probabilities,
    fit_per_attribute_temperatures,
)
from src.wacv.evaluation.distances import ranking_distance
from src.wacv.evaluation.predict import (
    predict_gallery_embeddings,
    predict_pooled_embeddings,
    predict_probabilities,
    unwrap_eval_module,
)
from src.wacv.evaluation.rerank import dual_softmax_rerank
from src.wacv.evaluation.smoothing import neighbour_smooth_probabilities
from src.wacv.monitoring.metrics import evaluate_ranking, mean_accuracy
from src.wacv.query_adapter import AttributeModelWithHardQuery


@dataclass(frozen=True)
class RankingEvalData:
    """Track-2 queries plus the gallery loader and its ground-truth attributes."""

    queries: np.ndarray
    gallery_attrs: np.ndarray
    gallery_loader: Iterable[tuple[torch.Tensor, torch.Tensor]]


def _write_calibration(path: Path, temperatures: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "per_attribute",
        "temperatures": [float(t) for t in temperatures.tolist()],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def rank_distances(
    queries: np.ndarray,
    probabilities: np.ndarray,
    ranking_config: RankingConfig,
    *,
    temperature: float,
    gallery_embeddings: np.ndarray | None = None,
    query_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """Configured ranking distance, then the optional dual-softmax query norm."""
    distances = ranking_distance(
        queries,
        probabilities,
        distance=ranking_config.distance,
        temperature=temperature,
        alpha=ranking_config.alpha,
        gallery_embeddings=gallery_embeddings,
        query_embeddings=query_embeddings,
        clear_fusion=ranking_config.clear_fusion,
    )
    if ranking_config.query_norm == "dual_softmax":
        distances = dual_softmax_rerank(
            distances, tau=float(ranking_config.query_norm_tau)
        )
    return distances


def compute_ranking_metrics(
    model: nn.Module,
    data: RankingEvalData,
    device: torch.device,
    ranking_config: RankingConfig,
    *,
    progress: bool,
    inference_config: InferenceConfig | None = None,
    calibrate: bool = False,
    calibration_out: Path | None = None,
) -> dict[str, float]:
    """Predict, rank, and score one gallery split.

    Shared by ``Trainer`` (per-epoch, EMA-wrapped model) and
    ``scripts/evaluate.py`` (standalone model). With ``calibrate`` set, one
    temperature per attribute is fitted on this split's own labels, applied
    before ranking, and written to ``calibration_out`` when given. Per-epoch
    training metrics leave it off so checkpoint selection never fits on labels.
    """
    inference = inference_config or InferenceConfig()
    probabilities = predict_probabilities(
        model,
        data.gallery_loader,
        device,
        progress=progress,
        tta=inference.tta,
        tta_scale=float(inference.tta_scale),
    )
    score_probabilities = probabilities
    if ranking_config.neighbour_smooth is not None:
        score_probabilities = neighbour_smooth_probabilities(
            probabilities,
            predict_pooled_embeddings(
                model, data.gallery_loader, device, progress=progress
            ),
            ranking_config.neighbour_smooth,
        )
    temperature = float(ranking_config.temperature)
    if calibrate:
        temperatures = fit_per_attribute_temperatures(
            score_probabilities, data.gallery_attrs
        )
        score_probabilities = calibrate_probabilities(score_probabilities, temperatures)
        temperature = 1.0
        if calibration_out is not None:
            _write_calibration(calibration_out, temperatures)

    gallery_embeddings: np.ndarray | None = None
    query_embeddings: np.ndarray | None = None
    if ranking_config.distance == "clear_hard":
        root = unwrap_eval_module(model)
        if not isinstance(root, AttributeModelWithHardQuery):
            raise TypeError("clear_hard requires AttributeModelWithHardQuery")
        gallery_embeddings = predict_gallery_embeddings(
            model, data.gallery_loader, device, progress=progress
        )
        query_tensor = torch.as_tensor(data.queries, dtype=torch.float32, device=device)
        with torch.inference_mode():
            query_embeddings = root.query_embeddings(query_tensor).cpu().numpy()

    distances = rank_distances(
        data.queries,
        score_probabilities,
        ranking_config,
        temperature=temperature,
        gallery_embeddings=gallery_embeddings,
        query_embeddings=query_embeddings,
    )
    metrics = evaluate_ranking(data.queries, data.gallery_attrs, distances)
    metrics["mA"] = mean_accuracy(probabilities, data.gallery_attrs)
    return metrics
