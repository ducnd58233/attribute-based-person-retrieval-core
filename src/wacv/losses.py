"""Training losses for attribute classifiers and query-aware ranking."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalBCEWithLogitsLoss(nn.Module):
    """Binary focal loss on logits, with optional positive-class weighting.

    When ``gamma == 0`` this matches ``BCEWithLogitsLoss`` (modulo reduction).
    """

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = float(gamma)
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.detach().float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal-modulated BCE from logits and soft/hard targets."""
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = (1.0 - p_t).clamp(min=0.0, max=1.0).pow(self.gamma) * bce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def build_attribute_criterion(
    *,
    focal_gamma: float,
    pos_weight: torch.Tensor,
) -> nn.Module:
    """Select BCE or focal-BCE from ``optimization.focal_gamma``."""
    weight = pos_weight.detach().float()
    if focal_gamma <= 0.0:
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    return FocalBCEWithLogitsLoss(gamma=focal_gamma, pos_weight=weight)


def hard_query_infonce(
    gallery_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """In-batch InfoNCE: each gallery row matches its own attribute query.

    ``gallery_embeddings`` and ``query_embeddings`` are L2-normalized
    ``(B, D)`` tensors. Diagonal pairs are positives.
    """
    logits = gallery_embeddings @ query_embeddings.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def pairwise_weighted_hamming(
    query_bits: torch.Tensor,
    attribute_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted Hamming distance matrix for binary query rows ``(Q, A)``."""
    if query_bits.ndim != 2:
        raise ValueError(f"query_bits must be 2D, got shape {tuple(query_bits.shape)}")
    weights = attribute_weights.float()
    if weights.shape != (query_bits.shape[1],):
        raise ValueError(
            "attribute_weights length must match query_bits num_attributes"
        )
    diff = (query_bits.unsqueeze(1) - query_bits.unsqueeze(0)).abs()
    return (diff * weights.view(1, 1, -1)).sum(dim=-1)


def asmr_semantic_margin_loss(
    gallery_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    local_query_index: torch.Tensor,
    margin_matrix: torch.Tensor,
) -> torch.Tensor:
    """Multi-prototype margin loss with per-pair semantic margins (ASMR-style)."""
    if gallery_embeddings.shape[0] == 0:
        return gallery_embeddings.new_zeros(())
    sim = gallery_embeddings @ query_embeddings.T
    labels = local_query_index.long()
    pos_sim = sim.gather(1, labels.unsqueeze(1))
    row_margins = margin_matrix.index_select(0, labels)
    mask = torch.ones_like(sim, dtype=torch.bool)
    mask.scatter_(1, labels.unsqueeze(1), False)
    violations = F.relu(sim - pos_sim + row_margins)
    selected = violations[mask]
    if selected.numel() == 0:
        return violations.new_zeros(())
    return selected.mean()


def logit_adjustment_offset(prior: torch.Tensor) -> torch.Tensor:
    """Return log(prior / (1 - prior)) per attribute (Menon et al. 2021)."""
    rates = prior.detach().float().clamp(1e-6, 1.0 - 1e-6)
    return torch.log(rates) - torch.log1p(-rates)


class LogitAdjustedLoss(nn.Module):
    """Apply a fixed logit offset during training, then delegate to ``base``.

    At evaluation the model emits raw logits (neutral prior), matching H10.
    """

    def __init__(self, base: nn.Module, offset: torch.Tensor) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("offset", offset.detach().float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """BCE (or focal) on ``logits + offset``."""
        return self.base(logits + self.offset, targets)


def bernoulli_nll_scores(
    queries: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """In-batch similarity: ``-Bernoulli-NLL(query_i, probs_j)`` as ``(B, B)``.

    Higher is better (Smooth-AP convention). ``queries`` and ``probabilities``
    are ``(B, A)``; probabilities should already be in ``(0, 1)``.
    """
    probs = probabilities.clamp(eps, 1.0 - eps)
    log_p = torch.log(probs)
    log_1mp = torch.log1p(-probs)
    # -Σ[q log p + (1-q) log(1-p)] == -q·(log p - log(1-p)) - 1·log(1-p)
    nll = -(queries @ (log_p - log_1mp).T) - log_1mp.sum(dim=1).unsqueeze(0)
    return -nll


def smooth_ap_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    tau: float = 0.01,
) -> torch.Tensor:
    """Differentiable AP surrogate (Brown et al., Smooth-AP, ECCV 2020).

    ``scores`` is an ``(B, B)`` similarity matrix (higher = more relevant).
    ``labels`` is ``(B,)`` identity / query-id labels; positives share a label.
    Returns ``1 - mean Smooth-AP`` over queries that have at least one positive.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"scores must be square (B, B), got {tuple(scores.shape)}")
    if labels.shape != (scores.shape[0],):
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match batch {scores.shape[0]}"
        )
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")

    batch_size = scores.shape[0]
    device = scores.device
    dtype = scores.dtype
    eye = torch.eye(batch_size, device=device, dtype=dtype)
    ones = torch.ones_like(eye)
    # soft[i, j, k] ≈ 1_{s_ik > s_ij}; exclude k∈{i, j}
    delta = scores.unsqueeze(1) - scores.unsqueeze(2)
    soft = torch.sigmoid(delta / tau)
    soft = soft * (ones - eye).unsqueeze(1) * (ones - eye).unsqueeze(0)

    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(dtype)
    positives = same * (ones - eye)
    rank_all = 1.0 + soft.sum(dim=2)
    rank_pos = 1.0 + (soft * positives.unsqueeze(1)).sum(dim=2)

    pos_counts = positives.sum(dim=1).clamp(min=1.0)
    ap = (rank_pos / rank_all * positives).sum(dim=1) / pos_counts
    has_pos = positives.sum(dim=1) > 0
    if not bool(has_pos.any()):
        return scores.new_zeros(())
    return 1.0 - ap[has_pos].mean()


def smooth_ap_aux_from_logits(
    logits: torch.Tensor,
    query_vectors: torch.Tensor,
    query_ids: torch.Tensor,
    *,
    tau: float = 0.01,
) -> torch.Tensor:
    """Smooth-AP on ``-NLL(query_vector_i, sigmoid(logits_j))`` with ``query_ids``."""
    probs = torch.sigmoid(logits)
    scores = bernoulli_nll_scores(query_vectors, probs)
    return smooth_ap_loss(scores, query_ids, tau=tau)
