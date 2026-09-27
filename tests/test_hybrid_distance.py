"""Tests for hybrid NLL+L1 ranking distance."""

from __future__ import annotations

import numpy as np

from src.wacv.evaluation import hybrid_distance, l1_distance, nll_distance


def test_hybrid_alpha_extremes() -> None:
    rng = np.random.default_rng(4)
    queries = rng.integers(0, 2, size=(3, 5)).astype(np.float32)
    probs = rng.random((6, 5), dtype=np.float32) * 0.8 + 0.1
    pure_nll = hybrid_distance(queries, probs, alpha=1.0, temperature=1.0)
    pure_l1 = hybrid_distance(queries, probs, alpha=0.0, temperature=1.0)

    def minmax(d: np.ndarray) -> np.ndarray:
        lo = d.min(axis=1, keepdims=True)
        hi = d.max(axis=1, keepdims=True)
        spread = hi - lo
        return np.where(spread > 0, (d - lo) / np.maximum(spread, 1e-8), 0.0)

    np.testing.assert_allclose(
        pure_nll, minmax(nll_distance(queries, probs, temperature=1.0)), atol=1e-5
    )
    np.testing.assert_allclose(pure_l1, minmax(l1_distance(queries, probs)), atol=1e-5)
