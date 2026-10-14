"""Vectorised ranking metrics must equal the plain per-query definitions."""

from __future__ import annotations

import numpy as np
import pytest

from src.wacv.monitoring import metrics
from src.wacv.monitoring.metrics import evaluate_ranking


def _reference_evaluate(
    queries: np.ndarray, gallery_attrs: np.ndarray, distances: np.ndarray
) -> dict[str, float]:
    """The earlier one-query-at-a-time implementation, kept as the oracle."""
    adm, ap, inp = [], [], []
    hits = {k: [] for k in (1, 5, 10)}
    for query, row in zip(queries, distances, strict=True):
        ranking = np.argsort(row, kind="stable")
        degree = (gallery_attrs == query).mean(axis=1)
        spread = degree.max() - degree.mean()
        ndom = (
            np.zeros_like(degree)
            if spread == 0
            else np.maximum(0.0, (degree - degree.mean()) / spread)
        )
        ranked_ndom = ndom[ranking]
        correct = np.flatnonzero(np.all(gallery_attrs[ranking] == query, axis=1)) + 1
        if correct.size == 0:
            adm.append(0.0)
            ap.append(0.0)
            inp.append(0.0)
            for flags in hits.values():
                flags.append(False)
            continue
        order = np.arange(1, correct.size + 1)
        ap.append(float(np.mean(order / correct)))
        adm.append(float((np.cumsum(ranked_ndom)[correct - 1] / correct).mean()))
        inp.append(float(correct.size / correct[-1]))
        for k, flags in hits.items():
            flags.append(bool(correct[0] <= k))
    return {
        "mADM": float(np.mean(adm)),
        "mAP": float(np.mean(ap)),
        **{f"Rank-{k}": float(np.mean(v)) for k, v in hits.items()},
        "mINP": float(np.mean(inp)),
    }


def _differences(actual: dict[str, float], expected: dict[str, float]) -> dict:
    """Metrics whose values differ by more than float rounding."""
    keys = actual.keys() | expected.keys()
    return {
        key: (actual.get(key), expected.get(key))
        for key in keys
        if actual.get(key) != pytest.approx(expected.get(key), rel=1e-9, abs=1e-12)
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_on_random_galleries(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(metrics, "QUERIES_PER_CHUNK", 7)  # uneven last chunk
    rng = np.random.default_rng(seed)
    gallery = (rng.random((300, 40)) < 0.3).astype(np.int64)
    queries = np.concatenate(
        [gallery[rng.choice(300, 15, replace=False)], rng.integers(0, 2, (4, 40))]
    )  # last four queries usually match nothing
    distances = rng.random((19, 300)).astype(np.float32)
    actual = evaluate_ranking(queries, gallery, distances)
    expected = _reference_evaluate(queries, gallery, distances)
    assert _differences(actual, expected) == {}


@pytest.mark.parametrize("max_matches_without_sort", [256, 1])
def test_ties_keep_gallery_order(
    max_matches_without_sort: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1 forces the sort fallback for every query with more than one match.
    monkeypatch.setattr(metrics, "MAX_MATCHES_WITHOUT_SORT", max_matches_without_sort)
    rng = np.random.default_rng(3)
    gallery = (rng.random((60, 40)) < 0.5).astype(np.int64)
    gallery[10:20] = gallery[5]  # repeated matches
    queries = gallery[[5, 30]]
    distances = np.round(rng.random((2, 60)), 1).astype(np.float32)  # many ties
    actual = evaluate_ranking(queries, gallery, distances)
    expected = _reference_evaluate(queries, gallery, distances)
    assert _differences(actual, expected) == {}


def test_constant_gallery_has_zero_ndom() -> None:
    gallery = np.zeros((5, 40), dtype=np.int64)
    queries = gallery[:1]
    distances = np.arange(5, dtype=np.float32)[None, :]
    result = evaluate_ranking(queries, gallery, distances)
    assert _differences(result, _reference_evaluate(queries, gallery, distances)) == {}
    assert result["mAP"] == 1.0
