"""Evaluate a run checkpoint on the Track 2 validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from src.wacv.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "convnext_baseline.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="defaults to runs/<run-name>/checkpoints/best.pt",
    )
    return parser.parse_args()


def _clear_hard_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    queries: np.ndarray,
    device: torch.device,
    *,
    progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    from src.wacv.evaluation import predict_gallery_embeddings
    from src.wacv.query_adapter import AttributeModelWithHardQuery

    if not isinstance(model, AttributeModelWithHardQuery):
        raise TypeError("clear_hard requires model.hard_query_adapter=true")
    gallery_embeddings = predict_gallery_embeddings(
        model, loader, device, progress=progress
    )
    query_tensor = torch.as_tensor(queries, dtype=torch.float32, device=device)
    with torch.inference_mode():
        query_embeddings = model.query_embeddings(query_tensor).cpu().numpy()
    return gallery_embeddings, query_embeddings


def _evaluate_checkpoint(
    config_path: Path, checkpoint: Path
) -> tuple[dict[str, float], ExperimentConfig]:
    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.checkpoint import load_model_weights
    from src.wacv.config import load_config
    from src.wacv.data import AttributeDataset, load_gt, load_queries
    from src.wacv.engine import resolve_device, resolve_num_workers
    from src.wacv.evaluation import (
        mean_accuracy,
        predict_probabilities,
        ranking_distance,
    )
    from src.wacv.metrics import evaluate_ranking
    from src.wacv.models import build_model

    config = load_config(config_path)
    device = resolve_device(config.run.device)
    val_root = config.data.root / "annotations" / "task2" / "val"
    ground_truth = load_gt(val_root / "gt.csv")
    attribute_names = attribute_names_from_columns(
        tuple(str(c) for c in ground_truth.columns)
    )
    queries = load_queries(val_root / "queries.csv")
    if config.data.limit is not None:
        ground_truth = ground_truth.iloc[: config.data.limit]
        queries = queries[: config.data.limit]

    workers = resolve_num_workers(config.data.num_workers)
    loader = DataLoader(
        AttributeDataset(
            ground_truth, config.data.root, attribute_names=attribute_names
        ),
        batch_size=config.data.batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    inference_model = config.model.model_copy(update={"pretrained": False})
    model = build_model(inference_model, num_attributes=len(attribute_names)).to(device)
    load_model_weights(checkpoint.resolve(), model, map_location=device)

    probabilities = predict_probabilities(
        model, loader, device, progress=config.run.progress
    )
    attributes = ground_truth[list(attribute_names)].to_numpy(dtype=np.int64)
    gallery_embeddings = None
    query_embeddings = None
    if config.ranking.distance == "clear_hard":
        gallery_embeddings, query_embeddings = _clear_hard_embeddings(
            model,
            loader,
            queries,
            device,
            progress=config.run.progress,
        )
    metrics = evaluate_ranking(
        queries,
        attributes,
        ranking_distance(
            queries,
            probabilities,
            distance=config.ranking.distance,
            temperature=config.ranking.temperature,
            gallery_embeddings=gallery_embeddings,
            query_embeddings=query_embeddings,
            clear_fusion=config.ranking.clear_fusion,
        ),
    )
    metrics["mA"] = mean_accuracy(probabilities, attributes)
    return metrics, config


def main() -> None:
    from src.wacv.config import load_config

    args = parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or (
        config.run.output_dir / config.run.name / "checkpoints" / "best.pt"
    )
    metrics, config = _evaluate_checkpoint(args.config, checkpoint)
    serialized = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    run_dir = config.run.output_dir / config.run.name
    if run_dir.is_dir():
        (run_dir / "evaluation.json").write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
