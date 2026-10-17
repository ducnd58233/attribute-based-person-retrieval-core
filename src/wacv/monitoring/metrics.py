"""The competition's ranking metrics: mADM, mAP, Rank-1/5/10, mINP.

Formulas: the parent workspace's PROBLEM-DETAILS.md, section 6. A gallery
image is a correct match for a query iff all 40 attributes are equal.
"""

from __future__ import annotations

import numpy as np

RANK_K_VALUES = (1, 5, 10)
QUERIES_PER_CHUNK = 64
# Above this many exact matches, one sort beats comparing every match
# against the whole gallery.
MAX_MATCHES_WITHOUT_SORT = 256


def _attribute_agreement(queries: np.ndarray, gallery_attrs: np.ndarray) -> np.ndarray:
    """``(Q, G)`` count of attributes on which each query and gallery row agree.

    Two matmuls on 0/1 values instead of comparing a ``(G, 40)`` matrix per
    query; counts are small integers, so float32 holds them exactly.
    """
    query_bits = queries.astype(np.float32)
    gallery_bits = gallery_attrs.astype(np.float32)
    agree_on_ones = query_bits @ gallery_bits.T
    agree_on_zeros = (1.0 - query_bits) @ (1.0 - gallery_bits).T
    return agree_on_ones + agree_on_zeros


def _normalized_degree_of_match(
    agreement: np.ndarray, num_attributes: int
) -> np.ndarray:
    """Rescale so the gallery mean is 0 and the max is 1, floored at 0."""
    degree = agreement.astype(np.float64) / num_attributes
    spread = degree.max() - degree.mean()
    if spread == 0:  # a gallery that all scores the same earns nothing
        return np.zeros_like(degree)
    return np.maximum(0.0, (degree - degree.mean()) / spread)


def _match_ranks_and_ndom_sums(
    distances: np.ndarray, ndom: np.ndarray, match_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """1-based rank of each match and the ndom summed over ranks 1..that rank.

    Ranking is ascending distance with ties in gallery order. Without sorting,
    the images ranked at or before match ``m`` are those with a smaller
    distance, or an equal distance and a gallery index no larger than ``m``.
    """
    if match_indices.size > MAX_MATCHES_WITHOUT_SORT:
        order = np.argsort(distances, kind="stable")
        rank_of = np.empty_like(order)
        rank_of[order] = np.arange(1, order.size + 1)
        cumulative_ndom = np.cumsum(ndom[order])
        ranks = rank_of[match_indices]
        return ranks, cumulative_ndom[ranks - 1]
    match_distances = distances[match_indices][:, None]
    gallery_index = np.arange(distances.size)[None, :]
    ranked_at_or_before = (distances[None, :] < match_distances) | (
        (distances[None, :] == match_distances)
        & (gallery_index <= match_indices[:, None])
    )
    return ranked_at_or_before.sum(axis=1), ranked_at_or_before @ ndom


def _score_query(
    distances: np.ndarray, agreement: np.ndarray, num_attributes: int
) -> tuple[float, float, float, int]:
    """(ADM, AP, INP, rank of the first match) for one query; first rank 0 = none."""
    match_indices = np.flatnonzero(agreement == num_attributes)
    if match_indices.size == 0:
        return 0.0, 0.0, 0.0, 0
    ndom = _normalized_degree_of_match(agreement, num_attributes)
    ranks, ndom_sums = _match_ranks_and_ndom_sums(distances, ndom, match_indices)
    by_rank = np.argsort(ranks)
    ranks, ndom_sums = ranks[by_rank], ndom_sums[by_rank]
    matches_so_far = np.arange(1, ranks.size + 1)
    adm = float(np.mean(ndom_sums / ranks))
    ap = float(np.mean(matches_so_far / ranks))
    inp = float(ranks.size / ranks[-1])
    return adm, ap, inp, int(ranks[0])


def evaluate_ranking(
    queries: np.ndarray,
    gallery_attrs: np.ndarray,
    distances: np.ndarray,
) -> dict[str, float]:
    """Score every query's ranking and average over queries.

    Args:
        queries: `(Q, 40)` binary attribute vectors.
        gallery_attrs: `(G, 40)` ground-truth attribute vectors, gallery order.
        distances: `(Q, G)`; row i ranks the gallery for `queries[i]`, smallest
            first (matches the competition's `"distances"` submission key).

    Returns:
        `{"mADM", "mAP", "Rank-1", "Rank-5", "Rank-10", "mINP"}`.

    Cost per query is O(G x matches) instead of a full O(G log G) sort, and
    attribute agreement is computed ``QUERIES_PER_CHUNK`` queries at a time.
    """
    num_queries, num_attributes = queries.shape
    scores = np.zeros((num_queries, 4))
    for start in range(0, num_queries, QUERIES_PER_CHUNK):
        agreement = _attribute_agreement(
            queries[start : start + QUERIES_PER_CHUNK], gallery_attrs
        )
        for offset, query_agreement in enumerate(agreement):
            scores[start + offset] = _score_query(
                distances[start + offset], query_agreement, num_attributes
            )
    adm, ap, inp, first_rank = scores.T
    found = first_rank > 0
    return {
        "mADM": float(adm.mean()),
        "mAP": float(ap.mean()),
        **{
            f"Rank-{k}": float((found & (first_rank <= k)).mean())
            for k in RANK_K_VALUES
        },
        "mINP": float(inp.mean()),
    }


def mean_accuracy(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Mean over attributes of balanced accuracy at a 0.5 threshold (mA)."""
    predictions = probabilities >= 0.5
    positives = targets == 1
    negatives = ~positives
    true_positive_rate = (predictions & positives).sum(axis=0) / np.maximum(
        positives.sum(axis=0), 1
    )
    true_negative_rate = ((~predictions) & negatives).sum(axis=0) / np.maximum(
        negatives.sum(axis=0), 1
    )
    return float((0.5 * (true_positive_rate + true_negative_rate)).mean())
