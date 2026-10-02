"""The competition's ranking metrics: mADM, mAP, Rank-1/5/10, mINP.

Formulas: the parent workspace's PROBLEM-DETAILS.md, section 6. A gallery
image is a correct match for a query iff all 40 attributes are equal.
"""

from __future__ import annotations

import numpy as np

RANK_K_VALUES = (1, 5, 10)


def _degree_of_match(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Share of the 40 attributes each gallery row agrees with the query on."""
    return (gallery == query).mean(axis=1)


def _normalized_dom(dom: np.ndarray) -> np.ndarray:
    """Rescale dom so the gallery mean sits at 0 and the max at 1, floored at 0."""
    spread = dom.max() - dom.mean()
    if spread == 0:
        return np.zeros_like(dom)
    return np.maximum(0.0, (dom - dom.mean()) / spread)


def _correct_ranks(query: np.ndarray, ranked_gallery: np.ndarray) -> np.ndarray:
    """1-indexed ranks, ascending, where ranked_gallery matches query exactly."""
    matches = np.all(ranked_gallery == query, axis=1)
    return np.flatnonzero(matches) + 1


def ap_score(correct_ranks: np.ndarray) -> float:
    """Average precision: rewards placing every correct match early."""
    if correct_ranks.size == 0:
        return 0.0
    hit_order = np.arange(1, correct_ranks.size + 1)
    return float(np.mean(hit_order / correct_ranks))


def inp_score(correct_ranks: np.ndarray) -> float:
    """Inverse negative penalty: how deep you must search for the last correct match."""
    if correct_ranks.size == 0:
        return 0.0
    return float(correct_ranks.size / correct_ranks[-1])


def adm_score(ranked_ndom: np.ndarray, correct_ranks: np.ndarray) -> float:
    """Average degree of match: like AP, but credits near-misses via ndom."""
    if correct_ranks.size == 0:
        return 0.0
    cumulative = np.cumsum(ranked_ndom)
    at_each_correct_rank = cumulative[correct_ranks - 1] / correct_ranks
    return float(at_each_correct_rank.mean())


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
    """
    num_queries = queries.shape[0]
    adm = np.zeros(num_queries)
    ap = np.zeros(num_queries)
    inp = np.zeros(num_queries)
    rank_hits = {k: np.zeros(num_queries, dtype=bool) for k in RANK_K_VALUES}

    for i in range(num_queries):
        query = queries[i]
        ranking = np.argsort(distances[i], kind="stable")
        ranked_attrs = gallery_attrs[ranking]
        ranked_ndom = _normalized_dom(_degree_of_match(query, gallery_attrs))[ranking]

        correct_ranks = _correct_ranks(query, ranked_attrs)
        adm[i] = adm_score(ranked_ndom, correct_ranks)
        ap[i] = ap_score(correct_ranks)
        inp[i] = inp_score(correct_ranks)
        if correct_ranks.size > 0:
            for k in RANK_K_VALUES:
                rank_hits[k][i] = correct_ranks[0] <= k

    return {
        "mADM": float(adm.mean()),
        "mAP": float(ap.mean()),
        **{f"Rank-{k}": float(rank_hits[k].mean()) for k in RANK_K_VALUES},
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
