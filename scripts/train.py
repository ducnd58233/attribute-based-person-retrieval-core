"""Train an attribute classifier from a validated YAML experiment config."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from src.wacv.config import ExperimentConfig
    from src.wacv.engine import RankingEvalData

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger(__name__)


def _join_task2_query_ids(
    train_gt: object,
    data_root: Path,
    *,
    limit: int | None,
) -> tuple[object, object, torch.Tensor]:
    """Inner-join task-2 train ids onto ``train_gt``; return gt, ids, query table."""
    import numpy as np
    import pandas as pd

    from src.wacv.data import load_queries, load_query_ids

    if not isinstance(train_gt, pd.DataFrame):
        raise TypeError("train_gt must be a DataFrame")
    task2_train = data_root / "annotations" / "task2" / "train"
    ids_path = task2_train / "ids.csv"
    queries_path = task2_train / "queries.csv"
    if not ids_path.is_file() or not queries_path.is_file():
        raise FileNotFoundError(
            f"smooth_ap_aux requires task-2 train ids/queries under {task2_train}"
        )
    ids_df = load_query_ids(ids_path)
    id_by_image = {
        str(row.image): int(row.query_id) for row in ids_df.itertuples(index=False)
    }
    joined = train_gt.loc[train_gt["image"].astype(str).isin(id_by_image)].reset_index(
        drop=True
    )
    if limit is not None:
        joined = joined.iloc[:limit].reset_index(drop=True)
    query_ids = joined["image"].astype(str).map(id_by_image).to_numpy(dtype=np.int64)
    query_table = torch.tensor(load_queries(queries_path), dtype=torch.float32)
    return joined, query_ids, query_table


def _attach_smooth_ap_if_needed(
    trainer: object,
    query_table: torch.Tensor | None,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> None:
    """Wire Smooth-AP query table onto ``trainer`` when aux weight is active."""
    if query_table is None:
        return
    from src.wacv.engine import Trainer

    if not isinstance(trainer, Trainer):
        raise TypeError("trainer must be a Trainer")
    trainer.attach_query_table(query_table)
    aux = config.losses.smooth_ap_aux
    logger.info(
        "smooth_ap_aux weight=%.3f tau=%.4f P=%d K=%d",
        aux.weight,
        aux.tau,
        aux.p,
        aux.k,
    )


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
) -> tuple[
    DataLoader, DataLoader, torch.Tensor, torch.Tensor, int, torch.Tensor | None
]:
    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.data import (
        AttributeDataset,
        QueryPkSampler,
        filter_gt_by_domains,
        load_gt,
        transforms_for_model,
    )
    from src.wacv.engine import resolve_num_workers

    annotation_root = config.data.root / "annotations" / "task1"
    train_gt = filter_gt_by_domains(
        load_gt(annotation_root / "train" / "gt.csv"), config.data.train_domains
    )
    val_gt = filter_gt_by_domains(
        load_gt(annotation_root / "val" / "gt.csv"), config.data.eval_domains
    )
    attribute_names = attribute_names_from_columns(
        tuple(str(c) for c in train_gt.columns)
    )
    if config.data.limit is not None:
        train_gt = train_gt.iloc[: config.data.limit]
        val_gt = val_gt.iloc[: config.data.limit]

    image_size = (int(config.data.image_size[0]), int(config.data.image_size[1]))
    train_transform = transforms_for_model(
        config.model.name,
        train=config.data.train_augmentation,
        image_size=image_size,
    )
    if not config.data.train_augmentation:
        train_transform = transforms_for_model(
            config.model.name, train=False, image_size=image_size
        )
    eval_transform = transforms_for_model(
        config.model.name, train=False, image_size=image_size
    )

    train_query_ids = None
    query_table: torch.Tensor | None = None
    aux = config.losses.smooth_ap_aux
    if aux.weight > 0.0:
        train_gt, train_query_ids, query_table = _join_task2_query_ids(
            train_gt, config.data.root, limit=config.data.limit
        )

    train_dataset = AttributeDataset(
        train_gt,
        config.data.root,
        train_transform,
        attribute_names=attribute_names,
        query_ids=train_query_ids,
    )
    val_dataset = AttributeDataset(
        val_gt, config.data.root, eval_transform, attribute_names=attribute_names
    )
    workers = resolve_num_workers(config.data.num_workers)
    pin = device.type == "cuda"
    if aux.weight > 0.0:
        assert train_query_ids is not None
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=QueryPkSampler(
                train_query_ids, p=aux.p, k=aux.k, seed=config.run.seed
            ),
            num_workers=workers,
            pin_memory=pin,
            persistent_workers=workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin,
            persistent_workers=workers > 0,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
    )

    positive_rates = train_gt[list(attribute_names)].astype(float).mean()
    positive_rates = positive_rates.clip(lower=1e-6)
    weights = ((1.0 - positive_rates) / positive_rates).clip(
        upper=config.optimization.pos_weight_clip
    )
    pos_weight = torch.tensor(weights.to_numpy(), dtype=torch.float32)
    prior = torch.tensor(positive_rates.to_numpy(), dtype=torch.float32)
    return (
        train_loader,
        val_loader,
        pos_weight,
        prior,
        len(attribute_names),
        query_table,
    )


def build_ranking_eval(
    config: ExperimentConfig, device: torch.device
) -> RankingEvalData | None:
    """Load Track-2 val queries/gallery for per-epoch ranking, or None if absent."""
    import numpy as np

    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.data import (
        AttributeDataset,
        load_gt,
        load_queries,
        transforms_for_model,
    )
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
            ground_truth,
            config.data.root,
            transforms_for_model(
                config.model.name,
                train=False,
                image_size=(
                    int(config.data.image_size[0]),
                    int(config.data.image_size[1]),
                ),
            ),
            attribute_names=attribute_names,
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


def maybe_wrap_logit_adjust(
    criterion: object,
    config: ExperimentConfig,
    prior: torch.Tensor,
    device: torch.device,
) -> object:
    """Optionally wrap the attribute criterion with a train-time prior offset."""
    if not config.optimization.logit_adjust:
        return criterion
    from src.wacv.losses import LogitAdjustedLoss, logit_adjustment_offset

    offset = logit_adjustment_offset(prior.to(device))
    LOGGER.info("logit_adjust enabled (train-time prior offset)")
    return LogitAdjustedLoss(criterion, offset)  # type: ignore[arg-type]


def main() -> None:
    from src.wacv.checkpoint import load_model_weights
    from src.wacv.config import load_config
    from src.wacv.engine import Trainer, resolve_device, seed_everything
    from src.wacv.losses import build_attribute_criterion
    from src.wacv.models import build_model
    from src.wacv.models.clip import ClipAttributeClassifier
    from src.wacv.monitoring.tracking import RunTracker

    config = load_config(parse_args().config)
    seed_everything(config.run.seed)
    device = resolve_device(config.run.device)
    if device.type == "cpu":
        LOGGER.warning(
            "no CUDA available; training on CPU (slow for ConvNeXt). "
            "On Colab: Runtime → Change runtime type → GPU."
        )
    train_loader, val_loader, pos_weight, prior, num_attributes, query_table = (
        build_loaders(config, device)
    )
    ranking = build_ranking_eval(config, device)

    model = build_model(config.model, num_attributes=num_attributes).to(device)
    if config.run.init_weights is not None:
        load_model_weights(
            config.run.init_weights.resolve(),
            model,
            map_location=device,
            strict=not config.model.hard_query_adapter,
        )
        LOGGER.info("initialized weights from %s", config.run.init_weights)
    criterion = build_attribute_criterion(
        focal_gamma=config.optimization.focal_gamma,
        pos_weight=pos_weight.to(device),
    )
    criterion = maybe_wrap_logit_adjust(criterion, config, prior, device)

    with RunTracker(config) as tracker:
        tracker.logger.info(
            "device=%s model=%s num_attributes=%d ranking_each_epoch=%s lp_ft=%s",
            device,
            config.model.name,
            num_attributes,
            ranking is not None,
            config.optimization.lp_ft,
        )
        if config.optimization.lp_ft and isinstance(model, ClipAttributeClassifier):
            model.freeze_backbone()
            lp_config = config.model_copy(
                update={
                    "optimization": config.optimization.model_copy(
                        update={"epochs": config.optimization.lp_epochs}
                    )
                }
            )
            lp_optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=config.optimization.lp_learning_rate,
                weight_decay=config.optimization.weight_decay,
            )
            lp_scheduler = ReduceLROnPlateau(
                lp_optimizer,
                factor=config.optimization.scheduler_factor,
                patience=config.optimization.scheduler_patience,
            )
            tracker.logger.info(
                "LP-FT stage 1: linear probe epochs=%d lr=%s",
                config.optimization.lp_epochs,
                config.optimization.lp_learning_rate,
            )
            Trainer(
                model, lp_optimizer, lp_scheduler, criterion, device, tracker, lp_config
            ).fit(train_loader, val_loader, ranking=ranking)
            model.unfreeze_backbone()
            ft_groups = model.layerwise_param_groups(
                learning_rate=config.optimization.learning_rate,
                weight_decay=config.optimization.weight_decay,
                layer_decay=config.optimization.layer_decay,
            )
            optimizer = torch.optim.AdamW(ft_groups)
            tracker.logger.info(
                "LP-FT stage 2: full FT epochs=%d base_lr=%s layer_decay=%s",
                config.optimization.epochs,
                config.optimization.learning_rate,
                config.optimization.layer_decay,
            )
        else:
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
        trainer = Trainer(
            model, optimizer, scheduler, criterion, device, tracker, config
        )
        _attach_smooth_ap_if_needed(
            trainer,
            None if query_table is None else query_table.to(device),
            config,
            tracker.logger,
        )
        result = trainer.fit(train_loader, val_loader, ranking=ranking)
        tracker.logger.info(
            "finished start_epoch=%d end_epoch=%d global_step=%d best_val_loss=%.6f",
            result.start_epoch,
            result.end_epoch,
            result.global_step,
            result.best_metric,
        )


if __name__ == "__main__":
    main()
