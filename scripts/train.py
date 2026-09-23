"""Train an attribute classifier from a validated YAML experiment config."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from src.wacv.config import ExperimentConfig
    from src.wacv.engine import RankingEvalData

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "convnext_baseline.yml",
    )
    return parser.parse_args()


def build_loaders(
    config: ExperimentConfig, device: torch.device
) -> tuple[DataLoader, DataLoader, torch.Tensor, int]:
    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.data import (
        IMAGE_TRANSFORM,
        TRAIN_TRANSFORM,
        AttributeDataset,
        load_gt,
    )
    from src.wacv.engine import resolve_num_workers

    annotation_root = config.data.root / "annotations" / "task1"
    train_gt = load_gt(annotation_root / "train" / "gt.csv")
    val_gt = load_gt(annotation_root / "val" / "gt.csv")
    attribute_names = attribute_names_from_columns(
        tuple(str(c) for c in train_gt.columns)
    )
    if config.data.limit is not None:
        train_gt = train_gt.iloc[: config.data.limit]
        val_gt = val_gt.iloc[: config.data.limit]

    train_transform = (
        TRAIN_TRANSFORM if config.data.train_augmentation else IMAGE_TRANSFORM
    )
    train_dataset = AttributeDataset(
        train_gt, config.data.root, train_transform, attribute_names=attribute_names
    )
    val_dataset = AttributeDataset(
        val_gt, config.data.root, IMAGE_TRANSFORM, attribute_names=attribute_names
    )
    workers = resolve_num_workers(config.data.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    positive_rates = train_gt[list(attribute_names)].astype(float).mean()
    positive_rates = positive_rates.clip(lower=1e-6)
    weights = ((1.0 - positive_rates) / positive_rates).clip(
        upper=config.optimization.pos_weight_clip
    )
    pos_weight = torch.tensor(weights.to_numpy(), dtype=torch.float32)
    return train_loader, val_loader, pos_weight, len(attribute_names)


def build_ranking_eval(
    config: ExperimentConfig, device: torch.device
) -> RankingEvalData | None:
    """Load Track-2 val queries/gallery for per-epoch ranking, or None if absent."""
    import numpy as np

    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.data import AttributeDataset, load_gt, load_queries
    from src.wacv.engine import RankingEvalData, resolve_num_workers

    val_root = config.data.root / "annotations" / "task2" / "val"
    gt_path = val_root / "gt.csv"
    queries_path = val_root / "queries.csv"
    if not gt_path.is_file() or not queries_path.is_file():
        LOGGER.warning(
            "Track-2 val annotations missing under %s; skipping per-epoch ranking",
            val_root,
        )
        return None

    ground_truth = load_gt(gt_path)
    attribute_names = attribute_names_from_columns(
        tuple(str(c) for c in ground_truth.columns)
    )
    queries = load_queries(queries_path)
    if config.data.limit is not None:
        ground_truth = ground_truth.iloc[: config.data.limit]
        queries = queries[: config.data.limit]

    workers = resolve_num_workers(config.data.num_workers)
    loader = DataLoader(
        AttributeDataset(
            ground_truth, config.data.root, attribute_names=attribute_names
        ),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    gallery_attrs = ground_truth[list(attribute_names)].to_numpy(dtype=np.int64)
    return RankingEvalData(
        queries=queries,
        gallery_attrs=gallery_attrs,
        gallery_loader=loader,
    )


def main() -> None:
    from src.wacv.config import load_config
    from src.wacv.engine import Trainer, resolve_device, seed_everything
    from src.wacv.models import build_model
    from src.wacv.tracking import RunTracker

    config = load_config(parse_args().config)
    seed_everything(config.run.seed)
    device = resolve_device(config.run.device)
    if device.type == "cpu":
        LOGGER.warning(
            "no CUDA available; training on CPU (slow for ConvNeXt). "
            "On Colab: Runtime → Change runtime type → GPU."
        )
    train_loader, val_loader, pos_weight, num_attributes = build_loaders(config, device)
    ranking = build_ranking_eval(config, device)

    model = build_model(config.model, num_attributes=num_attributes).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=config.optimization.scheduler_factor,
        patience=config.optimization.scheduler_patience,
    )

    with RunTracker(config) as tracker:
        tracker.logger.info(
            "device=%s model=%s num_attributes=%d ranking_each_epoch=%s",
            device,
            config.model.name,
            num_attributes,
            ranking is not None,
        )
        result = Trainer(
            model, optimizer, scheduler, criterion, device, tracker, config
        ).fit(train_loader, val_loader, ranking=ranking)
        tracker.logger.info(
            "finished start_epoch=%d end_epoch=%d global_step=%d best_val_loss=%.6f",
            result.start_epoch,
            result.end_epoch,
            result.global_step,
            result.best_metric,
        )


if __name__ == "__main__":
    main()
