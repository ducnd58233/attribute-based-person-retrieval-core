"""Tests for Bernoulli NLL ranking distance and submission parity."""

from __future__ import annotations

import numpy as np

from src.wacv.evaluation import nll_distance, ranking_distance


def test_nll_matches_naive_loop() -> None:
    rng = np.random.default_rng(1)
    queries = rng.integers(0, 2, size=(3, 5)).astype(np.float32)
    probs = rng.random((4, 5), dtype=np.float32) * 0.8 + 0.1
    via = nll_distance(queries, probs, temperature=1.0)
    eps = 1e-4
    clipped = np.clip(probs, eps, 1.0 - eps)
    naive = np.zeros((3, 4), dtype=np.float64)
    for i in range(3):
        for j in range(4):
            q = queries[i]
            p = clipped[j]
            naive[i, j] = -np.sum(q * np.log(p) + (1.0 - q) * np.log(1.0 - p))
    np.testing.assert_allclose(via, naive, rtol=1e-5, atol=1e-5)


def test_registry_nll_dispatch() -> None:
    rng = np.random.default_rng(2)
    queries = rng.integers(0, 2, size=(2, 4)).astype(np.float32)
    probs = rng.random((5, 4), dtype=np.float32)
    direct = nll_distance(queries, probs, temperature=1.5)
    via = ranking_distance(queries, probs, distance="nll", temperature=1.5)
    np.testing.assert_allclose(via, direct, rtol=0, atol=0)


def test_submission_nll_matches_library() -> None:
    import importlib.util
    from pathlib import Path

    run_path = Path(__file__).resolve().parents[1] / "submission" / "run.py"
    spec = importlib.util.spec_from_file_location("submission_run_nll", run_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(3)
    queries = rng.integers(0, 2, size=(2, 6)).astype(np.float32)
    probs = rng.random((3, 6), dtype=np.float32)
    lib = nll_distance(queries, probs, temperature=2.0)
    sub = mod._nll_distance(queries, probs, 2.0)
    np.testing.assert_allclose(sub, lib, rtol=1e-5, atol=1e-5)
