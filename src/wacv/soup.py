"""Greedy / uniform model soups (Wortsman et al., ICML 2022) — E22 / H22.

Averages compatible checkpoints that share initialisation. Greedy soup sorts
ingredients by a held-out metric (ID-val mADM), then adds a candidate only when
the uniform average of ``{soup ∪ candidate}`` does not hurt that metric.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

logger = logging.getLogger(__name__)

SoupMode = Literal["uniform", "greedy"]
ScoreFn = Callable[[Mapping[str, torch.Tensor]], float]


@dataclass(frozen=True)
class SoupIngredient:
    """One checkpoint participating in a soup."""

    name: str
    state_dict: Mapping[str, torch.Tensor]
    metric: float | None = None


@dataclass(frozen=True)
class SoupResult:
    """Averaged weights plus provenance for the selected ingredients."""

    state_dict: dict[str, torch.Tensor]
    selected: tuple[str, ...]
    mode: SoupMode
    ingredient_metrics: Mapping[str, float | None]


def _strip_module_prefix(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state):
        return dict(state)
    return {key.removeprefix("module."): value for key, value in state.items()}


def extract_model_state(
    checkpoint_path: Path, *, prefer_ema: bool = True
) -> dict[str, torch.Tensor]:
    """Load a trainer or raw state_dict checkpoint into a flat model state."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"unrecognised checkpoint type at {checkpoint_path}")
    if prefer_ema and isinstance(payload.get("ema_model"), dict):
        ema_state = {
            key: value
            for key, value in payload["ema_model"].items()
            if key != "n_averaged"
        }
        return _strip_module_prefix(ema_state)
    if isinstance(payload.get("model"), dict):
        return _strip_module_prefix(payload["model"])
    return _strip_module_prefix(payload)


def average_state_dicts(
    states: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Uniform mean of shared keys; non-floating tensors copy from the first."""
    if not states:
        raise ValueError("states must be non-empty")
    key_sets = [set(state) for state in states]
    shared = set.intersection(*key_sets)
    if not shared:
        raise ValueError("state dicts share no keys")
    only_in_some = set.union(*key_sets) - shared
    if only_in_some:
        logger.warning(
            "soup key mismatch: dropping %d keys (first: %s)",
            len(only_in_some),
            sorted(only_in_some)[:8],
        )
    merged: dict[str, torch.Tensor] = {}
    reference = states[0]
    for key in sorted(shared):
        tensors = [state[key] for state in states]
        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise ValueError(f"shape mismatch for {key}: {sorted(shapes)}")
        ref = reference[key]
        if not ref.is_floating_point():
            merged[key] = ref.clone()
            continue
        stacked = torch.stack([tensor.float() for tensor in tensors], dim=0)
        merged[key] = stacked.mean(dim=0).to(dtype=ref.dtype)
    return merged


def order_ingredients(
    ingredients: Sequence[SoupIngredient],
) -> list[SoupIngredient]:
    """Sort by metric descending; ``None`` metrics keep relative order at the end."""

    def sort_key(item: tuple[int, SoupIngredient]) -> tuple[int, float, int]:
        index, ingredient = item
        if ingredient.metric is None:
            return (1, 0.0, index)
        return (0, -float(ingredient.metric), index)

    ranked = sorted(enumerate(ingredients), key=sort_key)
    return [ingredient for _, ingredient in ranked]


def uniform_soup(ingredients: Sequence[SoupIngredient]) -> SoupResult:
    """Average every ingredient with equal weight."""
    if not ingredients:
        raise ValueError("ingredients must be non-empty")
    states = [ingredient.state_dict for ingredient in ingredients]
    return SoupResult(
        state_dict=average_state_dicts(states),
        selected=tuple(ingredient.name for ingredient in ingredients),
        mode="uniform",
        ingredient_metrics={
            ingredient.name: ingredient.metric for ingredient in ingredients
        },
    )


def greedy_soup(
    ingredients: Sequence[SoupIngredient],
    score_fn: ScoreFn,
) -> SoupResult:
    """Greedy soup: add candidates in metric order when the average does not hurt.

    ``score_fn`` should return a higher-is-better held-out score (e.g. ID-val
    mADM) for a candidate averaged state dict.
    """
    if not ingredients:
        raise ValueError("ingredients must be non-empty")
    ordered = order_ingredients(ingredients)
    selected = [ordered[0]]
    current_state = dict(ordered[0].state_dict)
    current_score = float(score_fn(current_state))
    for candidate in ordered[1:]:
        trial = [*selected, candidate]
        trial_state = average_state_dicts(
            [ingredient.state_dict for ingredient in trial]
        )
        trial_score = float(score_fn(trial_state))
        if trial_score >= current_score:
            selected = trial
            current_state = trial_state
            current_score = trial_score
    return SoupResult(
        state_dict=current_state,
        selected=tuple(ingredient.name for ingredient in selected),
        mode="greedy",
        ingredient_metrics={
            ingredient.name: ingredient.metric for ingredient in ingredients
        },
    )


def write_soup_checkpoint(
    result: SoupResult,
    output_path: Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Persist ``{"model": ...}`` plus soup metadata for packaging / evaluate."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model": result.state_dict,
        "soup": {
            "mode": result.mode,
            "selected": list(result.selected),
            "ingredient_metrics": {
                name: (None if value is None else float(value))
                for name, value in result.ingredient_metrics.items()
            },
        },
    }
    if extra:
        payload.update(dict(extra))
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)
    return output_path
