"""Sweep ranking-distance variants on one cached gallery probability matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _weighted_l1(
    queries: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    query_values = queries.astype(np.float32) * weights[None, :]
    predicted_values = probabilities.astype(np.float32)
    return (
        query_values.sum(axis=1, keepdims=True)
        + (predicted_values * weights[None, :]).sum(axis=1)[None, :]
        - 2.0 * query_values @ predicted_values.T
    )


def _load_or_predict_probabilities(
    ground_truth: object,
    attribute_names: tuple[str, ...],
    data_root: Path,
    batch_size: int,
    device: torch.device,
    cache: Path,
    checkpoint: Path,
) -> np.ndarray:
    from src.wacv.checkpoint import load_model_weights
    from src.wacv.config import load_config
    from src.wacv.data import AttributeDataset
    from src.wacv.evaluation import predict_probabilities
    from src.wacv.models import build_model

    if cache.is_file():
        return np.load(cache)

    config = load_config(ROOT / "configs" / "convnext_baseline.yml")
    loader = DataLoader(
        AttributeDataset(ground_truth, data_root, attribute_names=attribute_names),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_model(
        config.model.model_copy(update={"pretrained": False}),
        num_attributes=len(attribute_names),
    ).to(device)
    load_model_weights(checkpoint.resolve(), model, map_location=device)
    probabilities = predict_probabilities(model, loader, device, progress=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, probabilities)
    return probabilities


def _score_variants(
    queries: np.ndarray,
    attributes: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, dict[str, float]]:
    from src.wacv.evaluation import (
        _attribute_reliability_weights,
        _calibrate_probabilities,
        l1_distance,
        mean_accuracy,
        specker_distance,
    )
    from src.wacv.metrics import evaluate_ranking

    results: dict[str, dict[str, float]] = {}

    def score(name: str, distances: np.ndarray) -> None:
        metrics = evaluate_ranking(queries, attributes, distances)
        metrics["mA"] = mean_accuracy(probabilities, attributes)
        results[name] = metrics
        print(name, {key: round(value, 4) for key, value in metrics.items()})

    score("l1", l1_distance(queries, probabilities))
    for temperature in (0.75, 1.0, 1.25, 1.5, 2.0):
        score(
            f"specker_t{temperature}",
            specker_distance(queries, probabilities, temperature=temperature),
        )
    for temperature in (1.0, 1.5):
        calibrated = _calibrate_probabilities(probabilities, temperature)
        weights = np.ones(calibrated.shape[1], dtype=np.float32)
        score(f"temp_only_t{temperature}", _weighted_l1(queries, calibrated, weights))
    score(
        "weights_only",
        _weighted_l1(
            queries, probabilities, _attribute_reliability_weights(probabilities)
        ),
    )
    inverse_std = (1.0 / np.maximum(probabilities.std(axis=0), 1e-3)).astype(np.float32)
    inverse_std /= inverse_std.mean()
    score("invstd_weights", _weighted_l1(queries, probabilities, inverse_std))
    return results


def main() -> None:
    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.config import load_config
    from src.wacv.data import load_gt, load_queries
    from src.wacv.engine import resolve_device

    config = load_config(ROOT / "configs" / "convnext_baseline.yml")
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    val_root = config.data.root / "annotations" / "task2" / "val"
    ground_truth = load_gt(val_root / "gt.csv")
    attribute_names = attribute_names_from_columns(
        tuple(str(column) for column in ground_truth.columns)
    )
    queries = load_queries(val_root / "queries.csv")
    cache = ROOT / "runs" / "convnext-baseline" / "val_probs.npy"
    checkpoint = ROOT / "runs" / "convnext-baseline" / "best.pt"
    probabilities = _load_or_predict_probabilities(
        ground_truth,
        attribute_names,
        config.data.root,
        config.data.batch_size,
        device,
        cache,
        checkpoint,
    )
    attributes = ground_truth[list(attribute_names)].to_numpy(dtype=np.int64)
    results = _score_variants(queries, attributes, probabilities)
    output = ROOT / "runs" / "convnext-baseline" / "distance_sweep.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    best_name, best_metrics = max(results.items(), key=lambda item: item[1]["mADM"])
    print("BEST", best_name, best_metrics["mADM"])
    print("wrote", output)


if __name__ == "__main__":
    main()
