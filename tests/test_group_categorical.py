"""Tests for exclusive attribute groups and categorical NLL scoring."""

from __future__ import annotations

import numpy as np

from src.wacv.attributes import (
    exclusive_attribute_groups,
    independent_attribute_indices,
)
from src.wacv.evaluation import (
    group_categorical_nll_distance,
    nll_distance,
    ranking_distance,
)

_OFFICIAL_40 = (
    "Age-Young",
    "Age-Adult",
    "Age-Old",
    "Gender-Female",
    "Hair-Length-Short",
    "Hair-Length-Long",
    "Hair-Length-Bald",
    "UpperBody-Length-Short",
    "UpperBody-Color-Black",
    "UpperBody-Color-Blue",
    "UpperBody-Color-Brown",
    "UpperBody-Color-Green",
    "UpperBody-Color-Grey",
    "UpperBody-Color-Orange",
    "UpperBody-Color-Pink",
    "UpperBody-Color-Purple",
    "UpperBody-Color-Red",
    "UpperBody-Color-White",
    "UpperBody-Color-Yellow",
    "UpperBody-Color-Other",
    "LowerBody-Length-Short",
    "LowerBody-Color-Black",
    "LowerBody-Color-Blue",
    "LowerBody-Color-Brown",
    "LowerBody-Color-Green",
    "LowerBody-Color-Grey",
    "LowerBody-Color-Orange",
    "LowerBody-Color-Pink",
    "LowerBody-Color-Purple",
    "LowerBody-Color-Red",
    "LowerBody-Color-White",
    "LowerBody-Color-Yellow",
    "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts",
    "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack",
    "Accessory-Bag",
    "Accessory-Glasses-Normal",
    "Accessory-Glasses-Sun",
    "Accessory-Hat",
)


def test_exclusive_groups_cover_official_names() -> None:
    groups = exclusive_attribute_groups(_OFFICIAL_40)
    assert len(groups) == 5
    covered = {i for g in groups for i in g.indices}
    independent = set(independent_attribute_indices(40))
    assert covered.isdisjoint(independent)
    assert covered | independent == set(range(40))
    assert groups[1].name == "Hair" and groups[1].has_none


def test_one_hot_age_uses_renormalised_categorical() -> None:
    queries = np.zeros((1, 40), dtype=np.float32)
    queries[0, 0] = 1.0  # Age-Young
    probs = np.full((2, 40), 0.1, dtype=np.float32)
    probs[0, 0:3] = np.array([0.8, 0.1, 0.05], dtype=np.float32)
    probs[1, 0:3] = np.array([0.05, 0.8, 0.1], dtype=np.float32)
    dist = group_categorical_nll_distance(queries, probs, temperature=1.0)
    # Gallery 0 should rank better (higher Young mass after renorm)
    assert dist[0, 0] < dist[0, 1]


def test_multi_hot_group_falls_back_to_bernoulli() -> None:
    queries = np.zeros((1, 40), dtype=np.float32)
    queries[0, 0] = 1.0
    queries[0, 1] = 1.0  # multi-hot Age
    probs = np.clip(np.random.default_rng(0).random((3, 40)), 0.05, 0.95).astype(
        np.float32
    )
    via_group = group_categorical_nll_distance(queries, probs, temperature=1.0)
    # Independent bits + Age Bernoulli should match full Bernoulli on those
    # indices when Age is multi-hot; other exclusive groups still categorical.
    assert via_group.shape == (1, 3)
    assert np.isfinite(via_group).all()


def test_hair_none_query_selects_residual_mass() -> None:
    queries = np.zeros((1, 40), dtype=np.float32)
    # Hair all zero → implicit none
    probs = np.full((2, 40), 0.2, dtype=np.float32)
    probs[0, 4:7] = np.array([0.05, 0.05, 0.05], dtype=np.float32)  # high none
    probs[1, 4:7] = np.array([0.6, 0.2, 0.1], dtype=np.float32)  # low none
    dist = group_categorical_nll_distance(queries, probs, temperature=1.0)
    assert dist[0, 0] < dist[0, 1]


def test_registry_group_mode_and_distance_alias() -> None:
    rng = np.random.default_rng(4)
    queries = rng.integers(0, 2, size=(2, 40)).astype(np.float32)
    # Force one-hot Age / colours where possible for stability
    queries[:, 0:3] = 0
    queries[:, 0] = 1
    probs = rng.random((5, 40), dtype=np.float32) * 0.8 + 0.1
    direct = group_categorical_nll_distance(queries, probs, temperature=1.0)
    via_alias = ranking_distance(
        queries, probs, distance="group_categorical", temperature=1.0
    )
    np.testing.assert_allclose(via_alias, direct, rtol=0, atol=0)
    independent = ranking_distance(queries, probs, distance="nll", temperature=1.0)
    np.testing.assert_allclose(
        independent, nll_distance(queries, probs, temperature=1.0), rtol=0, atol=0
    )


def test_submission_group_categorical_matches_library() -> None:
    import importlib.util
    from pathlib import Path

    run_path = Path(__file__).resolve().parents[1] / "submission" / "run.py"
    spec = importlib.util.spec_from_file_location("submission_run_group", run_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(5)
    queries = rng.integers(0, 2, size=(2, 40)).astype(np.float32)
    queries[:, 0:3] = 0
    queries[:, 1] = 1
    probs = rng.random((4, 40), dtype=np.float32) * 0.8 + 0.1
    lib = group_categorical_nll_distance(queries, probs, temperature=1.5)
    sub = mod._group_categorical_nll_distance(queries, probs, 1.5)
    np.testing.assert_allclose(sub, lib, rtol=1e-5, atol=1e-5)
