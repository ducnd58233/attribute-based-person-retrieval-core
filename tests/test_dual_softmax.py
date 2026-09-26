"""Tests for dual-softmax query-set normalisation (E06)."""

from __future__ import annotations

import numpy as np

from src.wacv.evaluation import dual_softmax_rerank


def test_dual_softmax_preserves_shape_and_finite() -> None:
    rng = np.random.default_rng(0)
    distances = rng.random((20, 50), dtype=np.float32)
    out = dual_softmax_rerank(distances, tau=0.05)
    assert out.shape == distances.shape
    assert np.isfinite(out).all()


def test_dual_softmax_changes_ranking() -> None:
    distances = np.array(
        [[0.1, 5.0, 5.0], [5.0, 0.1, 5.0], [5.0, 5.0, 0.1]], dtype=np.float32
    )
    out = dual_softmax_rerank(distances, tau=0.05)
    # Diagonal should remain the preferred match after rerank
    assert out[0, 0] < out[0, 1]
    assert out[1, 1] < out[1, 0]


def test_submission_dual_softmax_matches_library() -> None:
    import importlib.util
    from pathlib import Path

    run_path = Path(__file__).resolve().parents[1] / "submission" / "run.py"
    spec = importlib.util.spec_from_file_location("submission_run_ds", run_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(3)
    distances = rng.random((8, 12), dtype=np.float32)
    lib = dual_softmax_rerank(distances, tau=0.1)
    sub = mod._dual_softmax_rerank(distances, 0.1)
    np.testing.assert_allclose(sub, lib, rtol=1e-5, atol=1e-5)
