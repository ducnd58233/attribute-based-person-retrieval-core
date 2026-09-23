"""Correctness tests for the competition's ranking metrics.

Every expected number here is hand-computed from the formulas in the parent
workspace's PROBLEM-DETAILS.md section 6, not copied from the
implementation, so a bug that made metrics.py internally consistent but
wrong would still be caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.wacv.metrics import evaluate_ranking


def test_perfect_ranking_scores_one() -> None:
    """Both exact matches ranked first (in either order) -> every metric is 1.0."""
    query = np.array([[1, 0, 0]])
    gallery = np.array(
        [
            [1, 0, 0],  # exact match, rank 1
            [1, 0, 0],  # exact match, rank 2
            [0, 1, 1],  # dom 0/3
            [1, 1, 0],  # dom 2/3
            [0, 0, 1],  # dom 1/3
        ]
    )
    distances = np.array([[0.0, 0.1, 0.9, 0.5, 0.7]])  # already best-first

    result = evaluate_ranking(query, gallery, distances)

    assert result["mADM"] == pytest.approx(1.0)
    assert result["mAP"] == pytest.approx(1.0)
    assert result["mINP"] == pytest.approx(1.0)
    assert result["Rank-1"] == pytest.approx(1.0)
    assert result["Rank-5"] == pytest.approx(1.0)


def test_hand_computed_single_query() -> None:
    """One query, five gallery images, one exact match at rank 3.

    Gallery in original order: g0=[1,0,0] (exact), g1=[1,0,1], g2=[0,0,0],
    g3=[1,1,1], g4=[0,1,1]. Distances rank them as g1, g3, g0, g4, g2, so
    the exact match lands at rank 3 of 5.

    dom = [1, 2/3, 2/3, 1/3, 0], mean = 8/15, max = 1, spread = 7/15.
    ndom = [1, 2/7, 2/7, 0, 0] (negative values floored at 0).
    Reordered to the ranking [g1, g3, g0, g4, g2]: ndom = [2/7, 0, 1, 0, 2/7].
    Cumulative at rank 3 = 2/7 + 0 + 1 = 9/7; ADM = (9/7) / 3 = 3/7.
    AP = INP = 1/3 (a single correct match at rank 3).
    """
    query = np.array([[1, 0, 0]])
    gallery = np.array(
        [
            [1, 0, 0],  # g0
            [1, 0, 1],  # g1
            [0, 0, 0],  # g2
            [1, 1, 1],  # g3
            [0, 1, 1],  # g4
        ]
    )
    distances = np.array([[0.5, 0.1, 0.9, 0.3, 0.7]])  # ranking: g1, g3, g0, g4, g2

    result = evaluate_ranking(query, gallery, distances)

    assert result["mADM"] == pytest.approx(3 / 7)
    assert result["mAP"] == pytest.approx(1 / 3)
    assert result["mINP"] == pytest.approx(1 / 3)
    assert result["Rank-1"] == pytest.approx(0.0)
    assert result["Rank-5"] == pytest.approx(1.0)


def test_worse_ranking_scores_lower_than_a_better_one() -> None:
    """Same query/gallery; pushing the exact match to last place scores worse."""
    query = np.array([[1, 0, 0]])
    gallery = np.array(
        [
            [1, 0, 0],  # exact match
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    better = evaluate_ranking(query, gallery, np.array([[0.1, 0.5, 0.6, 0.7, 0.9]]))
    worse = evaluate_ranking(query, gallery, np.array([[0.9, 0.5, 0.6, 0.7, 0.1]]))

    assert worse["mAP"] < better["mAP"]
    assert worse["mADM"] < better["mADM"]
    assert worse["mINP"] < better["mINP"]
    assert worse["Rank-1"] == 0.0
    assert better["Rank-1"] == 1.0
