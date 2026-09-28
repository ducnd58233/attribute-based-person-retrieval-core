"""MixStyle: channel-wise style mixing for domain generalization (Zhou et al.)."""

from __future__ import annotations

import torch
from torch import nn


class MixStyle(nn.Module):
    """Mix instance-level feature statistics across the batch.

    Active only in ``train()`` mode; ``eval()`` is an identity. With
    probability ``p``, replaces each sample's channel mean/std with a
    Beta(``alpha``, ``alpha``)-weighted blend against a random permutation
    of the batch (Zhou et al., MixStyle, ICLR 2021).
    """

    def __init__(
        self, *, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6
    ) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MixStyle to an NCHW feature map, or return ``x`` unchanged."""
        if not self.training or x.size(0) < 2:
            return x
        if torch.rand((), device=x.device).item() > self.p:
            return x

        batch_size = x.size(0)
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()
        x_normed = (x - mu) / sigma

        weight = (
            torch.distributions.Beta(self.alpha, self.alpha)
            .sample((batch_size, 1, 1, 1))
            .to(device=x.device, dtype=x.dtype)
        )
        perm = torch.randperm(batch_size, device=x.device)
        mu_mix = weight * mu + (1.0 - weight) * mu[perm]
        sigma_mix = weight * sigma + (1.0 - weight) * sigma[perm]
        return x_normed * sigma_mix + mu_mix
