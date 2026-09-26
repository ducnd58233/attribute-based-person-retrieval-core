"""Tests for per-attribute temperature calibration (E05)."""

from __future__ import annotations

import numpy as np

from src.wacv.evaluation import (
    apply_per_attribute_temperatures,
    fit_per_attribute_temperatures,
    nll_distance,
)


def test_fit_temperatures_shape_and_positive() -> None:
    rng = np.random.default_rng(0)
    probs = rng.random((50, 8), dtype=np.float32) * 0.8 + 0.1
    targets = (rng.random((50, 8)) > 0.5).astype(np.float32)
    temps = fit_per_attribute_temperatures(probs, targets, max_iter=20)
    assert temps.shape == (8,)
    assert np.all(temps > 0)


def test_apply_matches_scalar_broadcast() -> None:
    rng = np.random.default_rng(1)
    probs = rng.random((10, 5), dtype=np.float32) * 0.8 + 0.1
    temps = np.full(5, 2.0, dtype=np.float32)
    via_vec = apply_per_attribute_temperatures(probs, temps)
    # Scalar path through nll's calibrate: distance with T=2 on raw probs
    # should match nll on vector-calibrated probs at T=1
    d_scalar = nll_distance(np.ones((2, 5), dtype=np.float32), probs, temperature=2.0)
    d_vec = nll_distance(np.ones((2, 5), dtype=np.float32), via_vec, temperature=1.0)
    np.testing.assert_allclose(d_scalar, d_vec, rtol=1e-4, atol=1e-4)


def test_submission_apply_matches_library() -> None:
    import importlib.util
    from pathlib import Path

    run_path = Path(__file__).resolve().parents[1] / "submission" / "run.py"
    spec = importlib.util.spec_from_file_location("submission_run_cal", run_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(2)
    probs = rng.random((4, 6), dtype=np.float32) * 0.8 + 0.1
    temps = np.array([0.8, 1.0, 1.2, 1.5, 2.0, 0.5], dtype=np.float32)
    lib = apply_per_attribute_temperatures(probs, temps)
    sub = mod._apply_per_attribute_temperatures(probs, temps)
    np.testing.assert_allclose(sub, lib, rtol=1e-5, atol=1e-5)
