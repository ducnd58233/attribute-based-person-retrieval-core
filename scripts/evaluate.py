"""Evaluate a run checkpoint on the Track 2 validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

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


def main() -> None:
    from src.wacv.attributes import ATTRIBUTE_NAMES, NUM_ATTRIBUTES
    from src.wacv.checkpoint import load_model_weights
    from src.wacv.config import load_config
    from src.wacv.data import AttributeDataset, load_gt, load_queries
    from src.wacv.engine import resolve_device
    from src.wacv.evaluation import l1_distance, mean_accuracy, predict_probabilities
    from src.wacv.metrics import evaluate_ranking
    from src.wacv.models import build_model

    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config.run.device)
    checkpoint = args.checkpoint or (
        config.run.output_dir / config.run.name / "checkpoints" / "best.pt"
    )

    val_root = config.data.root / "annotations" / "task2" / "val"
    ground_truth = load_gt(val_root / "gt.csv")
    queries = load_queries(val_root / "queries.csv")
    if config.data.limit is not None:
        ground_truth = ground_truth.iloc[: config.data.limit]
        queries = queries[: config.data.limit]

    loader = DataLoader(
        AttributeDataset(ground_truth, config.data.root),
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
    )
    inference_model = config.model.model_copy(update={"pretrained": False})
    model = build_model(inference_model, num_attributes=NUM_ATTRIBUTES).to(device)
    load_model_weights(checkpoint.resolve(), model, map_location=device)

    probabilities = predict_probabilities(
        model, loader, device, progress=config.run.progress
    )
    attributes = ground_truth[list(ATTRIBUTE_NAMES)].to_numpy(dtype=np.int64)
    metrics = evaluate_ranking(queries, attributes, l1_distance(queries, probabilities))
    metrics["mA"] = mean_accuracy(probabilities, attributes)
    serialized = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    run_dir = config.run.output_dir / config.run.name
    if run_dir.is_dir():
        (run_dir / "evaluation.json").write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
