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
