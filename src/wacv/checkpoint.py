"""Complete, atomic training checkpoints for exact run continuation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel

from src.wacv.config import ExperimentConfig


@dataclass(frozen=True)
class ResumeState:
    next_epoch: int
    global_step: int
    best_metric: float


def _capture_rng_state() -> dict[str, object]:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": algorithm,
            "keys": keys.tolist(),
            "position": position,
            "has_gauss": has_gauss,
            "cached_gaussian": cached_gaussian,
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(cast(tuple[Any, ...], state["python"]))
    numpy_state = cast(dict[str, Any], state["numpy"])
    np.random.set_state(
        (
            numpy_state["algorithm"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch_state = cast(torch.Tensor, state["torch"])
    torch.set_rng_state(torch_state.cpu())
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = cast(list[torch.Tensor], state["cuda"])
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in cuda_states])


class CheckpointManager:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau,
        scaler: torch.amp.GradScaler,
        ema_model: AveragedModel,
        config: ExperimentConfig,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scaler = scaler
        self._ema_model = ema_model
        self._config = config

    def save(
        self, path: Path, *, epoch: int, global_step: int, best_metric: float
    ) -> None:
        payload = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "scaler": self._scaler.state_dict(),
            "ema_model": self._ema_model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "rng_state": _capture_rng_state(),
            "config": self._config.model_dump(mode="json"),
        }
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(path)

    def load(self, path: Path, *, map_location: torch.device) -> ResumeState:
        checkpoint = cast(
            dict[str, Any],
            torch.load(path, map_location=map_location, weights_only=True),
        )
        self._model.load_state_dict(checkpoint["model"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._scheduler.load_state_dict(checkpoint["scheduler"])
        self._scaler.load_state_dict(checkpoint["scaler"])
        self._ema_model.load_state_dict(checkpoint["ema_model"])
        _restore_rng_state(checkpoint["rng_state"])
        return ResumeState(
            next_epoch=int(checkpoint["epoch"]) + 1,
            global_step=int(checkpoint["global_step"]),
            best_metric=float(checkpoint["best_metric"]),
        )


def load_model_weights(
    path: Path, model: nn.Module, *, map_location: torch.device, prefer_ema: bool = True
) -> None:
    """Load a full trainer checkpoint or a legacy model-only state dict."""
    checkpoint = cast(
        dict[str, Any],
        torch.load(path, map_location=map_location, weights_only=True),
    )
    if "model" not in checkpoint:
        model.load_state_dict(checkpoint)
        return

    if prefer_ema and "ema_model" in checkpoint:
        ema_state = checkpoint["ema_model"]
        model_state = {
            key.removeprefix("module."): value
            for key, value in ema_state.items()
            if key != "n_averaged"
        }
    else:
        model_state = checkpoint["model"]
    model.load_state_dict(model_state)
