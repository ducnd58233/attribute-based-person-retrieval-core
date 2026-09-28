"""Unit tests for QueryPkSampler and Smooth-AP auxiliary loss (E16 / H16)."""

from __future__ import annotations

import torch

from src.wacv.data import QueryPkSampler
from src.wacv.losses import (
    bernoulli_nll_scores,
    smooth_ap_aux_from_logits,
    smooth_ap_loss,
)


def test_query_pk_sampler_shapes_and_grouping() -> None:
    # 4 identities with 3 images each; P=2, K=2 -> batch size 4
    query_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    sampler = QueryPkSampler(query_ids, p=2, k=2, seed=0)
    batches = list(sampler)
    assert len(batches) == 2
    for batch in batches:
        assert len(batch) == 4
        labels = [query_ids[i] for i in batch]
        counts: dict[int, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        assert len(counts) == 2
        assert all(count == 2 for count in counts.values())


def test_bernoulli_nll_scores_self_match_highest() -> None:
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    scores = bernoulli_nll_scores(queries, probs)
    assert scores.shape == (2, 2)
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]


def test_smooth_ap_loss_toy_batch_finite_and_nonnegative() -> None:
    # P=2, K=2 style batch: ids [0,0,1,1]
    labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor(
        [
            [1.0, 0.8, 0.1, 0.2],
            [0.7, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.9],
            [0.2, 0.1, 0.8, 1.0],
        ]
    )
    loss = smooth_ap_loss(scores, labels, tau=0.01)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0
    assert float(loss) <= 1.0


def test_smooth_ap_aux_from_logits_backward() -> None:
    logits = torch.randn(4, 3, requires_grad=True)
    query_vectors = torch.tensor(
        [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    query_ids = torch.tensor([0, 0, 1, 1])
    loss = smooth_ap_aux_from_logits(logits, query_vectors, query_ids, tau=0.01)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
