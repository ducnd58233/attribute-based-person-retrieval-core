"""Unit tests for SWAD start/stop scheduling (E14 / H14)."""

from __future__ import annotations

from src.wacv.swad import SwadController


def test_swad_starts_after_plateau_patience() -> None:
    ctrl = SwadController(plateau_patience=2, loss_tolerance=0.05)
    ctrl.observe_val_loss(1.0)
    assert not ctrl.should_average()
    ctrl.observe_val_loss(1.01)
    assert not ctrl.should_average()
    ctrl.observe_val_loss(1.02)
    assert ctrl.active
    assert ctrl.should_average()


def test_swad_stops_when_loss_exceeds_tolerance() -> None:
    ctrl = SwadController(plateau_patience=1, loss_tolerance=0.02)
    ctrl.observe_val_loss(1.0)
    ctrl.observe_val_loss(1.01)
    assert ctrl.should_average()
    ctrl.observe_val_loss(0.98)
    assert ctrl.should_average()
    ctrl.observe_val_loss(1.05)
    assert ctrl.stopped
    assert not ctrl.should_average()


def test_swad_improvement_resets_plateau_counter_before_start() -> None:
    ctrl = SwadController(plateau_patience=2, loss_tolerance=0.05)
    ctrl.observe_val_loss(1.0)
    ctrl.observe_val_loss(1.01)
    ctrl.observe_val_loss(0.95)
    assert not ctrl.active
    ctrl.observe_val_loss(0.96)
    assert not ctrl.active


def test_swad_state_round_trip() -> None:
    ctrl = SwadController(plateau_patience=2, loss_tolerance=0.01)
    ctrl.observe_val_loss(0.5)
    ctrl.observe_val_loss(0.51)
    ctrl.observe_val_loss(0.52)
    ctrl.mark_averaged()
    restored = SwadController.from_state_dict(
        ctrl.to_state_dict(),
        plateau_patience=2,
        loss_tolerance=0.01,
    )
    assert restored.active == ctrl.active
    assert restored.stopped == ctrl.stopped
    assert restored.ever_averaged
