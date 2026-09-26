"""Tests for logit-adjusted BCE (E10 / H10)."""

from __future__ import annotations

import torch
from torch import nn

from src.wacv.losses import LogitAdjustedLoss, logit_adjustment_offset


def test_logit_adjustment_offset_balanced() -> None:
    prior = torch.full((4,), 0.5)
    offset = logit_adjustment_offset(prior)
    assert torch.allclose(offset, torch.zeros(4), atol=1e-5)


def test_logit_adjusted_loss_shifts_logits() -> None:
    base = nn.BCEWithLogitsLoss(reduction="mean")
    prior = torch.tensor([0.9, 0.1])
    offset = logit_adjustment_offset(prior)
    wrapped = LogitAdjustedLoss(base, offset)
    logits = torch.zeros(2, 2)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    direct = base(logits + offset, targets)
    assert torch.allclose(wrapped(logits, targets), direct)
