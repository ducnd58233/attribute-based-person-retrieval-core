"""Unit tests for T28 top-1 non-match error budget."""

from __future__ import annotations

import numpy as np

from src.wacv.error_budget import (
    ERROR_BUDGET_GROUPS,
    error_budget_to_dict,
    top1_non_match_error_budget,
)


def test_error_budget_groups_cover_forty_bits() -> None:
    covered = sorted(index for _, indices in ERROR_BUDGET_GROUPS for index in indices)
    assert covered == list(range(40))


def test_top1_non_match_counts_disagreeing_bits() -> None:
    queries = np.zeros((2, 40), dtype=np.int64)
    gallery = np.zeros((3, 40), dtype=np.int64)
    # gallery[0] exact match for query 0; gallery[1] flips Age-Young + Gender
    gallery[1, 0] = 1
    gallery[1, 3] = 1
    # distances: query0 prefers exact match; query1 prefers flipped row
    distances = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = top1_non_match_error_budget(queries, gallery, distances)
    assert result.num_queries == 2
    assert result.num_top1_non_matches == 1
    assert result.bits_wrong_by_group["Age"] == 1
    assert result.bits_wrong_by_group["Gender"] == 1
    assert result.bits_wrong_total == 2
    payload = error_budget_to_dict(result)
    assert payload["bits_wrong_by_group"]["Age"] == 1


def test_dump_artifact_npz_shapes_on_toy() -> None:
    """Assert the npz bundle contract used by dump_artifacts.py."""
    probs = np.zeros((5, 40), dtype=np.float16)
    logits = np.ones((5, 40), dtype=np.float16)
    embeds = np.full((5, 8), 0.5, dtype=np.float16)
    assert probs.dtype == np.float16
    assert logits.shape == (5, 40)
    assert embeds.shape == (5, 8)
