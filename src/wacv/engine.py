"""Reusable training and validation loop for attribute classifiers."""

from __future__ import annotations

import importlib
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.wacv.checkpoint import CheckpointManager
from src.wacv.config import DeviceName, ExperimentConfig
from src.wacv.evaluation import l1_distance, mean_accuracy, predict_probabilities
from src.wacv.metrics import evaluate_ranking
from src.wacv.tracking import RunTracker

BatchIterable = Iterable[tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class FitResult:
    start_epoch: int
    end_epoch: int
    global_step: int
    best_metric: float


@dataclass(frozen=True)
class RankingEvalData:
    """Track-2 val queries + gallery for per-epoch ranking metrics."""

    queries: np.ndarray
    gallery_attrs: np.ndarray
    gallery_loader: BatchIterable


def resolve_device(requested: DeviceName) -> torch.device:
    """Resolve ``auto`` / ``cpu`` / ``cuda`` / ``tpu`` for notebooks and CLIs.

    ``auto``: CUDA, else ``torch_xla`` when importable, else CPU.
    """
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if requested in ("auto", "tpu"):
        try:
            xm = importlib.import_module("torch_xla.core.xla_model")
            return xm.xla_device()
        except Exception:
            if requested == "tpu":
                raise RuntimeError("TPU / torch_xla is not available") from None
            return torch.device("cpu")
    return torch.device(requested)


def resolve_num_workers(requested: int) -> int:
    """Cap DataLoader workers to ``cpu_count - 1`` (or 0 when requested ≤ 0)."""
    if requested <= 0:
        return 0
    cpu_count = os.cpu_count() or 1
    return min(requested, max(cpu_count - 1, 0))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """Own training state so fitting, checkpointing, and resume stay coherent."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau,
        criterion: nn.Module,
        device: torch.device,
        tracker: RunTracker,
        config: ExperimentConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.tracker = tracker
        self.config = config
        amp_enabled = config.optimization.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.ema_model = AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(config.optimization.ema_decay),
            use_buffers=True,
        )
        self.checkpoints = CheckpointManager(
            model, optimizer, scheduler, self.scaler, self.ema_model, config
        )

    def _train_epoch(self, loader: BatchIterable, epoch: int) -> tuple[float, int]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        global_step_delta = 0
        batches = tqdm(
            loader,
            desc=f"train {epoch + 1}/{self.config.optimization.epochs}",
            disable=not self.config.run.progress,
            leave=False,
        )
        use_amp = self.scaler.is_enabled()
        autocast_type = "cuda" if use_amp else "cpu"
        for images, targets in batches:
            batch_images = images.to(self.device, non_blocking=True)
            batch_targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_type, enabled=use_amp):
                smoothing = self.config.optimization.label_smoothing
                loss_targets = batch_targets * (1.0 - smoothing) + 0.5 * smoothing
                loss = self.criterion(self.model(batch_images), loss_targets)
            if use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if self.config.optimization.grad_clip_norm is not None:
                if use_amp:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimization.grad_clip_norm
                )
            if self.device.type == "xla":
                xm = importlib.import_module("torch_xla.core.xla_model")
                xm.optimizer_step(self.optimizer)
                self.ema_model.update_parameters(self.model)
            else:
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() >= scale_before:
                    self.ema_model.update_parameters(self.model)

            batch_size = batch_targets.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size
            global_step_delta += 1
            batches.set_postfix(loss=f"{loss.detach().item():.4f}")
        if total_samples == 0:
            raise ValueError("Training loader is empty")
        return total_loss / total_samples, global_step_delta

    def _validate_epoch(self, loader: BatchIterable, epoch: int) -> float:
        self.ema_model.eval()
        total_loss = 0.0
        total_samples = 0
        batches = tqdm(
            loader,
            desc=f"validate {epoch + 1}/{self.config.optimization.epochs}",
            disable=not self.config.run.progress,
            leave=False,
        )
        use_amp = self.scaler.is_enabled()
        autocast_type = "cuda" if use_amp else "cpu"
        with torch.inference_mode():
            for images, targets in batches:
                batch_images = images.to(self.device, non_blocking=True)
                batch_targets = targets.to(self.device, non_blocking=True)
                with torch.autocast(device_type=autocast_type, enabled=use_amp):
                    loss = self.criterion(self.ema_model(batch_images), batch_targets)
                batch_size = batch_targets.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        if total_samples == 0:
            raise ValueError("Validation loader is empty")
        return total_loss / total_samples

    def _ranking_metrics(self, ranking: RankingEvalData) -> dict[str, float]:
        probabilities = predict_probabilities(
            self.ema_model,
            ranking.gallery_loader,
            self.device,
            progress=False,
        )
        metrics = evaluate_ranking(
            ranking.queries,
            ranking.gallery_attrs,
            l1_distance(ranking.queries, probabilities),
        )
        metrics["mA"] = mean_accuracy(probabilities, ranking.gallery_attrs)
        return metrics

    @staticmethod
    def _update_best(
        current: float, best: float, higher_is_better: bool
    ) -> tuple[float, bool]:
        improved = current > best if higher_is_better else current < best
        return (current, True) if improved else (best, False)

    def _epoch_metrics(
        self,
        train_loss: float,
        val_loss: float,
        ranking: RankingEvalData | None,
    ) -> tuple[dict[str, float], dict[str, str], float]:
        metrics: dict[str, float] = {
            "train/loss": train_loss,
            "val/loss": val_loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        postfix: dict[str, str] = {
            "train": f"{train_loss:.4f}",
            "val": f"{val_loss:.4f}",
        }
        if ranking is None:
            return metrics, postfix, val_loss

        ranking_metrics = self._ranking_metrics(ranking)
        for key, value in ranking_metrics.items():
            metrics[f"val/{key}"] = value
        postfix["mAP"] = f"{ranking_metrics['mAP']:.4f}"
        postfix["R1"] = f"{ranking_metrics['Rank-1']:.4f}"
        # mADM is the competition's own ranking metric; select on it rather
        # than val_loss so best.pt tracks leaderboard quality instead of raw
        # BCE, which decouples from ranking quality once probabilities are
        # well-calibrated.
        selection_metric = ranking_metrics.get("mADM", ranking_metrics["mAP"])
        return metrics, postfix, selection_metric

    def fit(
        self,
        train_loader: BatchIterable,
        val_loader: BatchIterable,
        *,
        ranking: RankingEvalData | None = None,
    ) -> FitResult:
        start_epoch = 0
        global_step = 0
        select_higher_is_better = ranking is not None
        best_metric = float("-inf") if select_higher_is_better else float("inf")
        epochs_without_improvement = 0
        resume_path = self.config.run.resume
        if isinstance(resume_path, Path) and resume_path.is_file():
            state = self.checkpoints.load(resume_path, map_location=self.device)
            start_epoch = state.next_epoch
            global_step = state.global_step
            best_metric = state.best_metric
            self.tracker.logger.info(
                "resumed checkpoint=%s next_epoch=%d global_step=%d",
                resume_path,
                start_epoch,
                global_step,
            )
        elif resume_path is not None:
            self.tracker.logger.warning(
                "resume path missing (%s); starting from epoch 0",
                resume_path,
            )

        end_epoch = start_epoch - 1
        epoch_bar = tqdm(
            range(start_epoch, self.config.optimization.epochs),
            desc="epochs",
            disable=not self.config.run.progress,
            dynamic_ncols=True,
        )
        with logging_redirect_tqdm(loggers=[self.tracker.logger]):
            for epoch in epoch_bar:
                train_loss, step_delta = self._train_epoch(train_loader, epoch)
                global_step += step_delta
                val_loss = self._validate_epoch(val_loader, epoch)
                self.scheduler.step(val_loss)
                metrics, postfix, selection_metric = self._epoch_metrics(
                    train_loss, val_loss, ranking
                )
                best_metric, improved = self._update_best(
                    selection_metric, best_metric, select_higher_is_better
                )
                epochs_without_improvement = (
                    0 if improved else epochs_without_improvement + 1
                )
                epoch_bar.set_postfix(ordered_dict=dict(postfix))
                self.tracker.log_metrics(epoch, metrics)
                self.checkpoints.save(
                    self.tracker.checkpoint_dir / "last.pt",
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                )
                if improved:
                    self.checkpoints.save(
                        self.tracker.checkpoint_dir / "best.pt",
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=best_metric,
                    )
                end_epoch = epoch

                patience = self.config.optimization.early_stop_patience
                if patience is not None and epochs_without_improvement >= patience:
                    self.tracker.logger.info(
                        "early stopping: no improvement for %d epoch(s)", patience
                    )
                    break

        return FitResult(start_epoch, end_epoch, global_step, best_metric)
