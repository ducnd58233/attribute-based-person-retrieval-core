"""Offline parity: submission/run.py scorers vs src.wacv.evaluation.

Closes the E24 residual criterion that run.py and evaluation.py agree on the
S* stack (hybrid + dual-softmax) used by packaged identity-TTA zips.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

from src.wacv.evaluation import (
    dual_softmax_rerank,
    hybrid_distance,
    l1_distance,
    nll_distance,
    ranking_distance,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_PY_PATH = ROOT / "submission" / "run.py"


def _load_run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("submission_run_parity", RUN_PY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_queries_probs(
    rng: np.random.Generator, *, n_q: int, n_g: int, n_a: int
) -> tuple[np.ndarray, np.ndarray]:
    queries = rng.integers(0, 2, size=(n_q, n_a)).astype(np.float32)
    probs = rng.random((n_g, n_a), dtype=np.float32) * 0.8 + 0.1
    return queries, probs


def test_run_l1_matches_evaluation() -> None:
    run = _load_run_module()
    rng = np.random.default_rng(11)
    queries, probs = _random_queries_probs(rng, n_q=4, n_g=7, n_a=40)
    lib = l1_distance(queries, probs)
    sub = run._l1_distance(queries, probs)
    np.testing.assert_allclose(sub, lib, rtol=1e-5, atol=1e-5)


def test_run_nll_matches_evaluation() -> None:
    run = _load_run_module()
    rng = np.random.default_rng(12)
    queries, probs = _random_queries_probs(rng, n_q=3, n_g=5, n_a=40)
    for temperature in (1.0, 1.5, 2.0):
        lib = nll_distance(queries, probs, temperature=temperature)
        sub = run._nll_distance(queries, probs, temperature)
        np.testing.assert_allclose(sub, lib, rtol=1e-5, atol=1e-5)


def test_run_hybrid_matches_evaluation() -> None:
    run = _load_run_module()
    rng = np.random.default_rng(13)
    queries, probs = _random_queries_probs(rng, n_q=5, n_g=9, n_a=40)
    for alpha, temperature in ((0.5, 1.0), (0.25, 1.5), (0.75, 2.0)):
        lib = hybrid_distance(queries, probs, alpha=alpha, temperature=temperature)
        sub = run._hybrid_distance(queries, probs, temperature, alpha)
        np.testing.assert_allclose(sub, lib, rtol=1e-5, atol=1e-5)
        via_registry = ranking_distance(
            queries,
            probs,
            distance="hybrid",
            temperature=temperature,
            alpha=alpha,
        )
        np.testing.assert_allclose(via_registry, lib, rtol=0, atol=0)


def test_run_dual_softmax_matches_evaluation() -> None:
    run = _load_run_module()
    rng = np.random.default_rng(14)
    # Codabench-ish aspect ratio, still small enough for unit tests.
    distances = rng.random((64, 256), dtype=np.float32) * 5.0
    for tau in (0.05, 0.1):
        lib = dual_softmax_rerank(distances.copy(), tau=tau)
        sub = run._dual_softmax_rerank(distances.copy(), tau)
        np.testing.assert_allclose(sub, lib, rtol=1e-5, atol=1e-5)


def test_sstar_stack_parity_hybrid_then_dual_softmax() -> None:
    """E17/E20 package recipe: hybrid α=0.5 T=1.0 then dual-softmax τ=0.05."""
    run = _load_run_module()
    rng = np.random.default_rng(15)
    queries, probs = _random_queries_probs(rng, n_q=32, n_g=128, n_a=40)
    alpha, temperature, tau = 0.5, 1.0, 0.05

    lib_dist = hybrid_distance(queries, probs, alpha=alpha, temperature=temperature)
    lib_final = dual_softmax_rerank(lib_dist, tau=tau)

    sub_dist = run._hybrid_distance(queries, probs, temperature, alpha)
    sub_final = run._dual_softmax_rerank(sub_dist, tau)

    np.testing.assert_allclose(sub_final, lib_final, rtol=1e-5, atol=1e-5)
