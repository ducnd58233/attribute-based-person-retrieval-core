"""Output-space probability average of models that share one input transform."""

from __future__ import annotations

import torch
from torch import nn


class ProbabilityAverage(nn.Module):
    """Weighted mean of member sigmoid probabilities, returned as logits.

    Returning ``logit(mean(sigmoid(z_i)))`` keeps the usual logits contract, so
    every caller that applies ``sigmoid`` recovers the averaged probability.
    Members see the same input batch, so they must share an input transform.
    This averages outputs of separate forwards; it is not a weight soup.
    """

    def __init__(self, members: list[nn.Module], weights: list[float]) -> None:
        super().__init__()
        if len(members) != len(weights) or len(members) < 2:
            raise ValueError("need at least two members and one weight per member")
        if any(w < 0.0 for w in weights) or abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"weights must be >= 0 and sum to 1, got {weights}")
        self.members = nn.ModuleList(members)
        self.weights = tuple(float(w) for w in weights)
        # A buffer, not a tensor built in forward: torch.jit.trace would freeze
        # that tensor to the tracing device and break the package on CUDA.
        self.register_buffer(
            "member_weights",
            torch.tensor(self.weights, dtype=torch.float32).view(-1, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        probs = torch.stack(
            [torch.sigmoid(member(images).float()) for member in self.members]
        )
        weights = self.get_buffer("member_weights").float()
        return torch.logit((weights * probs).sum(dim=0), eps=1e-6)
