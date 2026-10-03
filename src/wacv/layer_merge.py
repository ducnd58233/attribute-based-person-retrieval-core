"""Layer-group coefficient merge for same-init CLIP checkpoints (E75 / H78)."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

LAYER_GROUPS: tuple[str, ...] = (
    "stem",
    "blocks_0_3",
    "blocks_4_7",
    "blocks_8_11",
    "ln_post_proj",
    "head",
)


def layer_group_for_key(key: str) -> str:
    """Map a CLIP AttributeModel state-dict key to a merge group."""
    if key.startswith("head") or key.startswith("classifier"):
        return "head"
    if key.startswith("visual.transformer.resblocks."):
        block_index = int(key.split(".")[3])
        if block_index <= 3:
            return "blocks_0_3"
        if block_index <= 7:
            return "blocks_4_7"
        return "blocks_8_11"
    if key.startswith("visual.ln_post") or key == "visual.proj":
        return "ln_post_proj"
    if key.startswith("visual."):
        return "stem"
    logger.warning("unknown key %s assigned to stem", key)
    return "stem"


def strip_module_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop a DistributedDataParallel ``module.`` prefix when present."""
    if not any(key.startswith("module.") for key in state):
        return dict(state)
    return {key.removeprefix("module."): value for key, value in state.items()}


def extract_model_state(
    checkpoint_path: str,
    *,
    prefer_ema: bool = True,
) -> dict[str, torch.Tensor]:
    """Load a trainer checkpoint into a flat floating model state dict."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"unrecognised checkpoint type: {type(payload)!r}")
    if prefer_ema and isinstance(payload.get("ema_model"), dict):
        ema_state = {
            key: value
            for key, value in payload["ema_model"].items()
            if key != "n_averaged"
        }
        return strip_module_prefix(ema_state)
    if isinstance(payload.get("model"), dict):
        return strip_module_prefix(payload["model"])
    return strip_module_prefix(payload)


def softmax_rows(logits: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax for ``[n_groups, n_members]`` coefficient logits."""
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=-1, keepdim=True)


@dataclass(frozen=True)
class LayerMergeResult:
    """Merged weights plus the per-group coefficient matrix used."""

    state_dict: dict[str, torch.Tensor]
    coefficients: tuple[tuple[float, ...], ...]
    group_names: tuple[str, ...]
    member_names: tuple[str, ...]


def merge_by_layer_groups(
    states: Sequence[Mapping[str, torch.Tensor]],
    member_names: Sequence[str],
    coeff_logits: torch.Tensor,
    *,
    group_names: Sequence[str] = LAYER_GROUPS,
) -> LayerMergeResult:
    """Softmax-weighted average of floating tensors within each layer group."""
    if len(states) < 2:
        raise ValueError("need at least two member states")
    if len(states) != len(member_names):
        raise ValueError("member_names length must match states")
    if coeff_logits.ndim != 2:
        raise ValueError("coeff_logits must be [n_groups, n_members]")
    if coeff_logits.shape != (len(group_names), len(states)):
        raise ValueError(
            f"coeff_logits shape {tuple(coeff_logits.shape)} != "
            f"({len(group_names)}, {len(states)})"
        )

    coefficients = softmax_rows(coeff_logits.float())
    key_sets = [set(state) for state in states]
    shared = set.intersection(*key_sets)
    if not shared:
        raise ValueError("state dicts share no keys")
    dropped = set.union(*key_sets) - shared
    if dropped:
        logger.warning(
            "layer merge dropping %d non-shared keys (first: %s)",
            len(dropped),
            sorted(dropped)[:8],
        )

    group_index = {name: index for index, name in enumerate(group_names)}
    merged: dict[str, torch.Tensor] = {}
    reference = states[0]
    for key in sorted(shared):
        ref = reference[key]
        if not torch.is_tensor(ref) or not ref.is_floating_point():
            merged[key] = ref.clone() if torch.is_tensor(ref) else ref
            continue
        group = layer_group_for_key(key)
        weights = coefficients[group_index[group]]
        stacked = torch.stack(
            [state[key].float() for state in states],
            dim=0,
        )
        shapes = {tuple(tensor.shape) for tensor in stacked}
        if len(shapes) != 1:
            raise ValueError(f"shape mismatch for {key}: {sorted(shapes)}")
        view = weights.view(-1, *([1] * (stacked.ndim - 1)))
        merged[key] = (stacked * view).sum(dim=0).to(dtype=ref.dtype)

    coeff_tuples = tuple(
        tuple(float(value) for value in row.tolist()) for row in coefficients
    )
    return LayerMergeResult(
        state_dict=merged,
        coefficients=coeff_tuples,
        group_names=tuple(group_names),
        member_names=tuple(member_names),
    )


def uniform_logits(n_groups: int, n_members: int) -> torch.Tensor:
    """Logits that softmax to a uniform average (all zeros)."""
    return torch.zeros(n_groups, n_members, dtype=torch.float32)


def decode_cma_vector(
    vector: Sequence[float],
    *,
    n_groups: int,
    n_members: int,
) -> torch.Tensor:
    """Map a flat CMA vector of length ``n_groups * n_members`` to logits."""
    expected = n_groups * n_members
    if len(vector) != expected:
        raise ValueError(f"vector length {len(vector)} != {expected}")
    return torch.tensor(list(vector), dtype=torch.float32).view(n_groups, n_members)


def cma_ask(
    mean: list[float],
    sigma: float,
    cov_diag: list[float],
    *,
    population: int,
    rng: torch.Generator,
) -> list[list[float]]:
    """Sample a diagonal-CMA population around ``mean``."""
    dim = len(mean)
    samples: list[list[float]] = []
    mean_t = torch.tensor(mean, dtype=torch.float32)
    scale = torch.tensor(cov_diag, dtype=torch.float32).clamp_min(1e-8).sqrt() * sigma
    for _ in range(population):
        noise = torch.randn(dim, generator=rng, dtype=torch.float32)
        sample = mean_t + scale * noise
        samples.append(sample.tolist())
    return samples


def cma_tell(
    mean: list[float],
    sigma: float,
    cov_diag: list[float],
    samples: Sequence[Sequence[float]],
    fitnesses: Sequence[float],
    *,
    learning_rate: float = 0.2,
) -> tuple[list[float], float, list[float]]:
    """Rank-μ update for diagonal CMA (Hansen tutorial, simplified)."""
    if len(samples) != len(fitnesses):
        raise ValueError("samples and fitnesses length mismatch")
    order = sorted(range(len(fitnesses)), key=lambda index: fitnesses[index])
    mu = max(1, len(samples) // 2)
    weights = [math.log(mu + 0.5) - math.log(rank + 1) for rank in range(mu)]
    weight_sum = sum(weights)
    weights = [weight / weight_sum for weight in weights]

    dim = len(mean)
    new_mean = [0.0] * dim
    for weight, index in zip(weights, order[:mu], strict=True):
        for axis, value in enumerate(samples[index]):
            new_mean[axis] += weight * value

    # Diagonal covariance from weighted residuals around the old mean.
    new_cov = list(cov_diag)
    for axis in range(dim):
        moment = 0.0
        for weight, index in zip(weights, order[:mu], strict=True):
            residual = samples[index][axis] - mean[axis]
            moment += weight * residual * residual
        target = moment / max(sigma * sigma, 1e-12)
        new_cov[axis] = (1.0 - learning_rate) * cov_diag[axis] + learning_rate * target
        new_cov[axis] = min(max(new_cov[axis], 1e-6), 10.0)

    # Step-size adapts gently to ranked progress.
    best = fitnesses[order[0]]
    median = fitnesses[order[len(order) // 2]]
    sigma = min(sigma * 1.05, 2.0) if best < median else max(sigma * 0.95, 0.05)

    blended_mean = [
        (1.0 - learning_rate) * old + learning_rate * updated
        for old, updated in zip(mean, new_mean, strict=True)
    ]
    return blended_mean, sigma, new_cov
