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
    from src.wacv.engine import AsmrTrainContext, DistillTrainContext, TrainerAuxContext
    from src.wacv.evaluation import RankingEvalData
    from src.wacv.models.clip import ClipAttributeClassifier
    from src.wacv.models.dinov2 import Dinov2AttributeClassifier
    from src.wacv.monitoring.tracking import RunTracker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    from src.wacv.config import DEFAULT_CONFIG_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    return parser.parse_args()


def build_asmr_supervision(
    config: ExperimentConfig,
) -> tuple[dict[str, int] | None, AsmrTrainContext | None]:
    from src.wacv.data import load_task2_train_query_supervision
    from src.wacv.engine import AsmrTrainContext

    if config.losses.asmr_aux.weight <= 0.0:
        return None, None
    mapping, queries = load_task2_train_query_supervision(config.data.root)
    context = AsmrTrainContext(
        train_query_vectors=torch.from_numpy(queries.astype("float32"))
    )
    return mapping, context


def build_loaders(
    config: ExperimentConfig,
    device: torch.device,
    *,
    query_id_by_image: dict[str, int] | None = None,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, int]:
    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.data import (
        AttributeDataset,
        build_loader,
        filter_gt_by_domains,
        load_gt,
        transforms_for_model,
    )

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

    train_transform = transforms_for_model(
        config.model.name,
        train=config.data.train_augmentation,
        image_size=config.data.image_size,
        degrade_aug=config.data.degrade_aug,
    )
    eval_transform = transforms_for_model(
        config.model.name, train=False, image_size=config.data.image_size
    )
    train_dataset = AttributeDataset(
        train_gt,
        config.data.root,
        train_transform,
        attribute_names=attribute_names,
        query_id_by_image=query_id_by_image,
    )
    val_dataset = AttributeDataset(
        val_gt, config.data.root, eval_transform, attribute_names=attribute_names
    )
    train_loader = build_loader(train_dataset, config.data, device, shuffle=True)
    val_loader = build_loader(val_dataset, config.data, device, shuffle=False)

    positive_rates = train_gt[list(attribute_names)].astype(float).mean()
    positive_rates = positive_rates.clip(lower=1e-6)
    weights = ((1.0 - positive_rates) / positive_rates).clip(
        upper=config.optimization.pos_weight_clip
    )
    pos_weight = torch.tensor(weights.to_numpy(), dtype=torch.float32)
    prior = torch.tensor(positive_rates.to_numpy(), dtype=torch.float32)
    return train_loader, val_loader, pos_weight, prior, len(attribute_names)


def build_ranking_eval(
    config: ExperimentConfig, device: torch.device
) -> RankingEvalData | None:
    """Load Track-2 val queries/gallery for per-epoch ranking, or None if absent."""
    import numpy as np

    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.data import (
        AttributeDataset,
        build_loader,
        load_gt,
        load_queries,
        transforms_for_model,
    )
    from src.wacv.evaluation import RankingEvalData

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

    dataset = AttributeDataset(
        ground_truth,
        config.data.root,
        transforms_for_model(
            config.model.name, train=False, image_size=config.data.image_size
        ),
        attribute_names=attribute_names,
    )
    loader = build_loader(dataset, config.data, device, shuffle=False)
    gallery_attrs = ground_truth[list(attribute_names)].to_numpy(dtype=np.int64)
    return RankingEvalData(
        queries=queries,
        gallery_attrs=gallery_attrs,
        gallery_loader=loader,
    )


def maybe_wrap_logit_adjust(
    criterion: torch.nn.Module,
    config: ExperimentConfig,
    prior: torch.Tensor,
    device: torch.device,
) -> torch.nn.Module:
    if not config.optimization.logit_adjust:
        return criterion
    from src.wacv.losses import LogitAdjustedLoss, logit_adjustment_offset

    offset = logit_adjustment_offset(prior.to(device))
    LOGGER.info("logit_adjust enabled (train-time prior offset)")
    return LogitAdjustedLoss(criterion, offset)


def build_distill_context(
    config: ExperimentConfig,
    *,
    num_attributes: int,
    device: torch.device,
) -> DistillTrainContext | None:
    """Load a frozen teacher for soft+hard KD, or None when distill is off."""
    from src.wacv.checkpoint import load_model_weights
    from src.wacv.config import MODEL_NATIVE_IMAGE_SIZE, ModelConfig
    from src.wacv.engine import DistillTrainContext
    from src.wacv.models import build_model

    distill = config.losses.distill
    if distill.soft_weight <= 0.0:
        return None
    teacher_cfg = distill.teacher
    if teacher_cfg is None:
        raise ValueError("losses.distill.soft_weight > 0 requires teacher")
    checkpoint = teacher_cfg.checkpoint
    if checkpoint is None:
        raise ValueError("losses.distill.teacher.checkpoint is required")
    image_size = teacher_cfg.image_size or MODEL_NATIVE_IMAGE_SIZE[teacher_cfg.name]
    teacher_model_config = ModelConfig(
        name=teacher_cfg.name,
        pretrained=teacher_cfg.pretrained,
        dropout=teacher_cfg.dropout,
        freeze_backbone=teacher_cfg.freeze_backbone,
        lora_rank=teacher_cfg.lora_rank,
        lora_alpha=teacher_cfg.lora_alpha,
    )
    teacher = build_model(teacher_model_config, num_attributes=num_attributes)
    load_model_weights(checkpoint.resolve(), teacher, map_location=device, strict=True)
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    LOGGER.info(
        "distill teacher=%s checkpoint=%s image_size=%s "
        "soft_weight=%s hard_weight=%s T=%s",
        teacher_cfg.name,
        checkpoint,
        image_size,
        distill.soft_weight,
        distill.hard_weight,
        distill.temperature,
    )
    return DistillTrainContext(teacher=teacher, teacher_image_size=image_size)


def _run_linear_probe_stage(
    *,
    model: ClipAttributeClassifier | Dinov2AttributeClassifier,
    criterion: torch.nn.Module,
    device: torch.device,
    tracker: RunTracker,
    config: ExperimentConfig,
    fit_data: tuple[
        DataLoader,
        DataLoader,
        RankingEvalData | None,
        TrainerAuxContext,
    ],
) -> torch.optim.Optimizer:
    """Stage-1 LP then build the stage-2 AdamW optimizer with layer-wise LR."""
    from src.wacv.engine import Trainer

    train_loader, val_loader, ranking, aux = fit_data
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
        model,
        lp_optimizer,
        lp_scheduler,
        criterion,
        device,
        tracker,
        lp_config,
        aux=aux,
    ).fit(train_loader, val_loader, ranking=ranking)
    model.unfreeze_backbone()
    ft_groups = model.layerwise_param_groups(
        learning_rate=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
        layer_decay=config.optimization.layer_decay,
    )
    tracker.logger.info(
        "LP-FT stage 2: full FT epochs=%d base_lr=%s layer_decay=%s",
        config.optimization.epochs,
        config.optimization.learning_rate,
        config.optimization.layer_decay,
    )
    return torch.optim.AdamW(ft_groups)


def main() -> None:
    from src.wacv.checkpoint import load_model_weights
    from src.wacv.config import load_config
    from src.wacv.engine import Trainer, resolve_device, seed_everything
    from src.wacv.losses import build_attribute_criterion
    from src.wacv.models import build_model
    from src.wacv.models.clip import ClipAttributeClassifier
    from src.wacv.models.dinov2 import Dinov2AttributeClassifier
    from src.wacv.monitoring.tracking import RunTracker
    from src.wacv.utils import configure_default_logging

    configure_default_logging()
    config = load_config(parse_args().config)
    seed_everything(config.run.seed)
    device = resolve_device(config.run.device)
    if device.type == "cpu":
        LOGGER.warning(
            "no CUDA available; training on CPU (slow for ConvNeXt). "
            "On Colab: Runtime → Change runtime type → GPU."
        )
    query_id_by_image, asmr_context = build_asmr_supervision(config)
    train_loader, val_loader, pos_weight, prior, num_attributes = build_loaders(
        config, device, query_id_by_image=query_id_by_image
    )
    ranking = build_ranking_eval(config, device)
    distill_context = build_distill_context(
        config, num_attributes=num_attributes, device=device
    )
    from src.wacv.engine import TrainerAuxContext

    aux = TrainerAuxContext(asmr=asmr_context, distill=distill_context)

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
            "device=%s model=%s num_attributes=%d "
            "ranking_each_epoch=%s lp_ft=%s distill=%s",
            device,
            config.model.name,
            num_attributes,
            ranking is not None,
            config.optimization.lp_ft,
            config.losses.distill.soft_weight > 0.0,
        )
        if config.optimization.lp_ft and isinstance(
            model, ClipAttributeClassifier | Dinov2AttributeClassifier
        ):
            optimizer = _run_linear_probe_stage(
                model=model,
                criterion=criterion,
                device=device,
                tracker=tracker,
                config=config,
                fit_data=(train_loader, val_loader, ranking, aux),
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
        result = Trainer(
            model,
            optimizer,
            scheduler,
            criterion,
            device,
            tracker,
            config,
            aux=aux,
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
