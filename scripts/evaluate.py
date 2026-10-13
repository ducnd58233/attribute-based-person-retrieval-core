"""Evaluate a run checkpoint on the Track 2 validation split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from src.wacv.config import EvalProtocol, ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METRIC_KEY_MAP = {
    "mADM": "madm",
    "mAP": "map",
    "Rank-1": "rank1",
    "Rank-5": "rank5",
    "Rank-10": "rank10",
    "mINP": "minp",
    "mA": "ma",
}


def parse_args() -> argparse.Namespace:
    from src.wacv.config import DEFAULT_CONFIG_PATH, DistanceName

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="defaults to runs/<run-name>/checkpoints/best.pt",
    )
    parser.add_argument(
        "--protocol",
        choices=("id_val", "lodo_m", "both"),
        default="both",
        help="id_val uses official queries; lodo_m uses Market1501 holdout",
    )
    parser.add_argument(
        "--distance",
        choices=get_args(DistanceName),
        default=None,
        help="override ranking.distance from the config",
    )
    parser.add_argument(
        "--ensemble-checkpoint",
        type=Path,
        default=None,
        help="second checkpoint whose probabilities are averaged with the primary",
    )
    parser.add_argument(
        "--ensemble-config",
        type=Path,
        default=None,
        help="config for --ensemble-checkpoint (same model input as the primary)",
    )
    parser.add_argument(
        "--ensemble-weight",
        type=float,
        default=0.5,
        help="weight on the primary model probabilities; the rest goes to the second",
    )
    return parser.parse_args()


def _protocol_metrics_keys(
    raw: dict[str, float], wall_seconds: float
) -> dict[str, float]:
    out: dict[str, float] = {}
    for source, target in METRIC_KEY_MAP.items():
        if source in raw:
            out[target] = float(raw[source])
    out["wall_seconds"] = float(wall_seconds)
    return out


def _evaluate_protocol(
    *,
    config: ExperimentConfig,
    model: torch.nn.Module,
    device: torch.device,
    ground_truth: pd.DataFrame,
    queries: np.ndarray,
    attribute_names: tuple[str, ...],
    distance: str,
    calibration_out: Path | None = None,
) -> dict[str, float]:
    from src.wacv.data import build_eval_loader
    from src.wacv.evaluation import RankingEvalData, compute_ranking_metrics

    data = RankingEvalData(
        queries=queries,
        gallery_attrs=ground_truth[list(attribute_names)].to_numpy(dtype=np.int64),
        gallery_loader=build_eval_loader(
            ground_truth, attribute_names, config.model.name, config.data, device
        ),
    )
    started = time.perf_counter()
    metrics = compute_ranking_metrics(
        model,
        data,
        device,
        config.ranking.model_copy(update={"distance": distance}),
        progress=config.run.progress,
        inference_config=config.inference,
        calibrate=config.ranking.calibration == "per_attribute",
        calibration_out=calibration_out,
    )
    return _protocol_metrics_keys(metrics, time.perf_counter() - started)


def _prepare_protocol_data(
    config: ExperimentConfig,
    protocol: EvalProtocol,
) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.data import (
        build_lodo_m_split,
        filter_gt_by_domains,
        load_gt,
        load_queries,
    )

    val_root = config.data.root / "annotations" / "task2" / "val"
    ground_truth = load_gt(val_root / "gt.csv")
    attribute_names = attribute_names_from_columns(
        tuple(str(c) for c in ground_truth.columns)
    )
    if protocol == "lodo_m":
        ground_truth, queries = build_lodo_m_split(ground_truth, attribute_names)
    else:
        ground_truth = filter_gt_by_domains(ground_truth, config.data.eval_domains)
        queries = load_queries(val_root / "queries.csv")
    if config.data.limit is not None:
        ground_truth = ground_truth.iloc[: config.data.limit].reset_index(drop=True)
        queries = queries[: config.data.limit]
    return ground_truth, queries, attribute_names


def evaluate_checkpoint(
    config_path: Path,
    checkpoint: Path,
    *,
    protocols: tuple[EvalProtocol, ...],
    distance: str | None = None,
    ensemble_config_path: Path | None = None,
    ensemble_checkpoint: Path | None = None,
    ensemble_weight: float = 0.5,
) -> tuple[dict[str, dict[str, float]], ExperimentConfig]:
    from src.wacv.config import load_config
    from src.wacv.engine import resolve_device
    from src.wacv.models.ensemble import ProbabilityAverage

    config = load_config(config_path)
    device = resolve_device(config.run.device)
    distance_name = distance or config.ranking.distance

    _, _, attribute_names = _prepare_protocol_data(config, protocols[0])
    model = _load_eval_model(config, checkpoint, len(attribute_names), device)
    if ensemble_checkpoint is not None:
        if ensemble_config_path is None:
            raise ValueError("--ensemble-checkpoint requires --ensemble-config")
        other_config = load_config(ensemble_config_path)
        if (other_config.model.name, other_config.data.image_size) != (
            config.model.name,
            config.data.image_size,
        ):
            raise ValueError("ensemble members must share model name and image size")
        other = _load_eval_model(
            other_config, ensemble_checkpoint, len(attribute_names), device
        )
        model = ProbabilityAverage(
            [model, other], [ensemble_weight, 1.0 - ensemble_weight]
        ).to(device)

    results: dict[str, dict[str, float]] = {}
    run_dir = config.run.output_dir / config.run.name
    for protocol in protocols:
        ground_truth, queries, names = _prepare_protocol_data(config, protocol)
        calibration_path = None
        if config.ranking.calibration == "per_attribute":
            calibration_path = run_dir / f"calibration_{protocol}.json"
            if protocol == "id_val":
                calibration_path = run_dir / "calibration.json"
        results[protocol] = _evaluate_protocol(
            config=config,
            model=model,
            device=device,
            ground_truth=ground_truth,
            queries=queries,
            attribute_names=names,
            distance=distance_name,
            calibration_out=calibration_path,
        )
    return results, config


def _load_eval_model(
    config: ExperimentConfig,
    checkpoint: Path,
    num_attributes: int,
    device: torch.device,
) -> torch.nn.Module:
    from src.wacv.checkpoint import load_model_weights
    from src.wacv.models import build_model

    inference_model = config.model.model_copy(update={"pretrained": False})
    model = build_model(inference_model, num_attributes=num_attributes)
    # Trainer checkpoints carry optimizer and EMA tensors; deserialize them on
    # CPU so two L-scale members do not exhaust GPU memory before load.
    load_model_weights(checkpoint.resolve(), model, map_location=torch.device("cpu"))
    return model.to(device)


def _merge_metrics_file(
    path: Path, update: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            for key, value in existing.items():
                if isinstance(value, dict):
                    merged[key] = {str(k): float(v) for k, v in value.items()}
    merged.update(update)
    return merged


def main() -> None:
    from src.wacv.config import load_config
    from src.wacv.utils import configure_default_logging

    configure_default_logging()
    args = parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or (
        config.run.output_dir / config.run.name / "checkpoints" / "best.pt"
    )
    protocols: tuple[EvalProtocol, ...] = (
        ("id_val", "lodo_m") if args.protocol == "both" else (args.protocol,)
    )

    results, config = evaluate_checkpoint(
        args.config,
        checkpoint,
        protocols=protocols,
        distance=args.distance,
        ensemble_config_path=args.ensemble_config,
        ensemble_checkpoint=args.ensemble_checkpoint,
        ensemble_weight=args.ensemble_weight,
    )
    serialized = json.dumps(results, indent=2, sort_keys=True) + "\n"
    run_dir = config.run.output_dir / config.run.name
    if run_dir.is_dir():
        metrics_path = run_dir / "metrics.json"
        merged = _merge_metrics_file(metrics_path, results)
        metrics_path.write_text(
            json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(serialized, end="")


if __name__ == "__main__":
    main()
