"""SWAD (dense stochastic weight averaging) schedule for validation-loss windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


@dataclass
class SwadController:
    """Start at ID-val loss plateau; stop when loss exceeds min plus tolerance."""

    plateau_patience: int
    loss_tolerance: float
    active: bool = False
    stopped: bool = False
    epochs_without_improvement: int = 0
    best_val_loss: float = float("inf")
    min_loss_while_active: float = float("inf")
    ever_averaged: bool = False

    def observe_val_loss(self, val_loss: float) -> None:
        """Update plateau / SWAD window state after one validation epoch."""
        if self.stopped:
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if not self.active and self.epochs_without_improvement >= self.plateau_patience:
            self.active = True
            self.min_loss_while_active = val_loss

        if self.active:
            self.min_loss_while_active = min(self.min_loss_while_active, val_loss)
            if val_loss > self.min_loss_while_active + self.loss_tolerance:
                self.active = False
                self.stopped = True

    def should_average(self) -> bool:
        """Whether the current training step should update the running average."""
        return self.active and not self.stopped

    def mark_averaged(self) -> None:
        self.ever_averaged = True

    def to_state_dict(self) -> dict[str, bool | int | float]:
        return {
            "active": self.active,
            "stopped": self.stopped,
            "epochs_without_improvement": self.epochs_without_improvement,
            "best_val_loss": self.best_val_loss,
            "min_loss_while_active": self.min_loss_while_active,
            "ever_averaged": self.ever_averaged,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        plateau_patience: int,
        loss_tolerance: float,
    ) -> SwadController:
        return cls(
            plateau_patience=plateau_patience,
            loss_tolerance=loss_tolerance,
            active=bool(state["active"]),
            stopped=bool(state["stopped"]),
            epochs_without_improvement=int(state["epochs_without_improvement"]),
            best_val_loss=float(state["best_val_loss"]),
            min_loss_while_active=float(state["min_loss_while_active"]),
            ever_averaged=bool(state.get("ever_averaged", False)),
        )


def swad_state_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    """Return SWAD controller state if present in a trainer checkpoint."""
    raw = checkpoint.get("swad")
    if raw is None:
        return None
    return cast(dict[str, Any], raw)
