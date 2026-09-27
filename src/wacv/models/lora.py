"""Low-rank adapters for frozen Linear layers (no peft dependency)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization


class _LoRAWeight(nn.Module):
    """Parametrization: ``W' = W + (alpha/r) B A`` so F.linear sees LoRA."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Return base weight plus low-rank residual."""
        return weight + (self.lora_B @ self.lora_A) * self.scaling


def inject_lora_into_linear(module: nn.Linear, *, rank: int, alpha: float) -> nn.Linear:
    """Attach LoRA to ``module.weight`` via parametrization; freeze base W/b.

    PyTorch ``MultiheadAttention`` reads ``out_proj.weight`` with ``F.linear``,
    so a wrapper Module that only overrides ``forward`` is skipped. Parametrizing
    the weight keeps that path while training only A/B.
    """
    if rank < 1:
        raise ValueError(f"LoRA rank must be >= 1, got {rank}")
    module.weight.requires_grad = False
    if module.bias is not None:
        module.bias.requires_grad = False
    register_parametrization(
        module,
        "weight",
        _LoRAWeight(
            module.in_features,
            module.out_features,
            rank=rank,
            alpha=alpha,
        ),
    )
    return module
