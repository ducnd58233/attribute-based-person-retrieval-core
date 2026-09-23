"""Train an attribute classifier from a validated YAML experiment config."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.wacv.attributes import ATTRIBUTE_NAMES, NUM_ATTRIBUTES
from src.wacv.config import ExperimentConfig, load_config
from src.wacv.data import (
    IMAGE_TRANSFORM,
    TRAIN_TRANSFORM,
    AttributeDataset,
    load_gt,
)
from src.wacv.engine import Trainer, resolve_device, seed_everything
from src.wacv.models import build_model
from src.wacv.tracking import RunTracker


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
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    annotation_root = config.data.root / "annotations" / "task1"
    train_gt = load_gt(annotation_root / "train" / "gt.csv")
    val_gt = load_gt(annotation_root / "val" / "gt.csv")
    if config.data.limit is not None:
        train_gt = train_gt.iloc[: config.data.limit]
        val_gt = val_gt.iloc[: config.data.limit]

    train_transform = (
        TRAIN_TRANSFORM if config.data.train_augmentation else IMAGE_TRANSFORM
    )
    train_dataset = AttributeDataset(train_gt, config.data.root, train_transform)
    val_dataset = AttributeDataset(val_gt, config.data.root, IMAGE_TRANSFORM)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
    )

    positive_rates = train_gt[list(ATTRIBUTE_NAMES)].astype(float).mean()
    positive_rates = positive_rates.clip(lower=1e-6)
    weights = ((1.0 - positive_rates) / positive_rates).clip(
        upper=config.optimization.pos_weight_clip
    )
    pos_weight = torch.tensor(weights.to_numpy(), dtype=torch.float32)
    return train_loader, val_loader, pos_weight


def main() -> None:
    config = load_config(parse_args().config)
    seed_everything(config.run.seed)
    device = resolve_device(config.run.device)
    train_loader, val_loader, pos_weight = build_loaders(config, device)

    model = build_model(config.model, num_attributes=NUM_ATTRIBUTES).to(device)
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
        tracker.logger.info("device=%s model=%s", device, config.model.name)
        result = Trainer(
            model, optimizer, scheduler, criterion, device, tracker, config
        ).fit(train_loader, val_loader)
        tracker.logger.info(
            "finished start_epoch=%d end_epoch=%d global_step=%d best_val_loss=%.6f",
            result.start_epoch,
            result.end_epoch,
            result.global_step,
            result.best_metric,
        )


if __name__ == "__main__":
    main()
