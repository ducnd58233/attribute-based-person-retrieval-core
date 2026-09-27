"""Low-rank adapters for frozen Linear layers (no peft dependency)."""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """``y = W x + b + (alpha/r) B A x`` with frozen base weights."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        in_features = base.in_features
        out_features = base.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply base Linear then LoRA residual."""
        base_out = self.base(inputs)
        # (..., in) @ (in, r) -> (..., r) @ (r, out) via A^T then B^T
        lora_out = (inputs @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return base_out + lora_out * self.scaling


def inject_lora_into_linear(
    module: nn.Linear, *, rank: int, alpha: float
) -> LoRALinear:
    """Wrap a Linear as LoRALinear (callers must reassign the attribute)."""
    return LoRALinear(module, rank=rank, alpha=alpha)
