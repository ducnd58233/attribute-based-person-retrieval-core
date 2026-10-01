"""Top-1 non-match attribute error budget for T28 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# TASKS groups covering all 40 bits (length bits folded into colour groups).
ERROR_BUDGET_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("Age", tuple(range(0, 3))),
    ("Gender", (3,)),
    ("Hair", tuple(range(4, 7))),
    ("Upper colour", tuple(range(7, 20))),
    ("Lower colour", tuple(range(20, 33))),
    ("Lower type", tuple(range(33, 35))),
    ("Accessories", tuple(range(35, 40))),
)


@dataclass(frozen=True)
class ErrorBudgetResult:
    """Aggregated bit-disagreement counts for top-1 non-matches."""

    num_queries: int
    num_top1_non_matches: int
    bits_wrong_by_group: dict[str, int]
    bits_wrong_total: int


def top1_non_match_error_budget(
    queries: np.ndarray,
    gallery_attrs: np.ndarray,
    distances: np.ndarray,
) -> ErrorBudgetResult:
    """Count bit disagreements for each query's top-1 ranked non-match.

    A gallery row is a full match when all 40 attributes equal the query. When
    the nearest gallery image (smallest distance) is not a full match, each
    disagreeing bit increments its attribute-group counter.
    """
    if queries.ndim != 2 or gallery_attrs.ndim != 2:
        raise ValueError("queries and gallery_attrs must be 2-D")
    if distances.shape != (queries.shape[0], gallery_attrs.shape[0]):
        raise ValueError(
            f"distances shape {distances.shape} != "
            f"({queries.shape[0]}, {gallery_attrs.shape[0]})"
        )
    group_counts = {name: 0 for name, _ in ERROR_BUDGET_GROUPS}
    non_matches = 0
    total_bits = 0
    for query_index in range(queries.shape[0]):
        query = queries[query_index]
        top1 = int(np.argmin(distances[query_index]))
        gallery_row = gallery_attrs[top1]
        if np.all(gallery_row == query):
            continue
        non_matches += 1
        disagree = gallery_row != query
        for name, indices in ERROR_BUDGET_GROUPS:
            wrong = int(np.count_nonzero(disagree[list(indices)]))
            group_counts[name] += wrong
            total_bits += wrong
    return ErrorBudgetResult(
        num_queries=int(queries.shape[0]),
        num_top1_non_matches=non_matches,
        bits_wrong_by_group=group_counts,
        bits_wrong_total=total_bits,
    )


def error_budget_to_dict(result: ErrorBudgetResult) -> dict[str, object]:
    """Serialize an error budget for metrics.json."""
    return {
        "num_queries": result.num_queries,
        "num_top1_non_matches": result.num_top1_non_matches,
        "bits_wrong_total": result.bits_wrong_total,
        "bits_wrong_by_group": dict(result.bits_wrong_by_group),
    }
