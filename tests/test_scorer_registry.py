"""Regression: registry dispatch matches direct scorer functions."""

from __future__ import annotations

import numpy as np

from src.wacv.evaluation import (
    clear_hard_distance,
    l1_distance,
    ranking_distance,
    specker_distance,
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_registry_l1_matches_direct() -> None:
    rng = _rng()
    queries = rng.integers(0, 2, size=(5, 8)).astype(np.float32)
    probs = rng.random((11, 8), dtype=np.float32)
    direct = l1_distance(queries, probs)
    via = ranking_distance(queries, probs, distance="l1")
    np.testing.assert_allclose(via, direct, rtol=0, atol=0)


def test_registry_specker_matches_direct() -> None:
    rng = _rng()
    queries = rng.integers(0, 2, size=(4, 6)).astype(np.float32)
    probs = rng.random((9, 6), dtype=np.float32)
    direct = specker_distance(queries, probs, temperature=1.5)
    via = ranking_distance(queries, probs, distance="specker", temperature=1.5)
    np.testing.assert_allclose(via, direct, rtol=0, atol=0)


def test_registry_clear_hard_matches_direct() -> None:
    rng = _rng()
    queries = rng.integers(0, 2, size=(3, 5)).astype(np.float32)
    probs = rng.random((7, 5), dtype=np.float32)
    gallery_emb = rng.standard_normal((7, 4)).astype(np.float32)
    gallery_emb /= np.linalg.norm(gallery_emb, axis=1, keepdims=True) + 1e-8
    query_emb = rng.standard_normal((3, 4)).astype(np.float32)
    query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8
    direct = clear_hard_distance(
        queries,
        probs,
        gallery_emb,
        query_emb,
        temperature=1.5,
        clear_fusion=0.4,
    )
    via = ranking_distance(
        queries,
        probs,
        distance="clear_hard",
        temperature=1.5,
        gallery_embeddings=gallery_emb,
        query_embeddings=query_emb,
        clear_fusion=0.4,
    )
    np.testing.assert_allclose(via, direct, rtol=0, atol=0)
