"""Unit tests for WiSE-FT interpolation helpers (E18)."""

from __future__ import annotations

import torch

from src.wacv.wise_ft import (
    attribute_display_phrase,
    interpolate_state_dicts,
    positive_prompt,
)


def test_attribute_display_phrase() -> None:
    assert attribute_display_phrase("Age-Young") == "age young"
    assert "age young" in positive_prompt("Age-Young")


def test_interpolate_state_dicts_endpoints() -> None:
    zs = {"w": torch.zeros(2, 3), "b": torch.ones(2)}
    ft = {"w": torch.ones(2, 3), "b": torch.full((2,), 3.0)}
    at_zero = interpolate_state_dicts(zs, ft, alpha=0.0)
    at_one = interpolate_state_dicts(zs, ft, alpha=1.0)
    at_half = interpolate_state_dicts(zs, ft, alpha=0.5)
    assert torch.allclose(at_zero["w"], zs["w"])
    assert torch.allclose(at_one["w"], ft["w"])
    assert torch.allclose(at_half["w"], torch.full((2, 3), 0.5))
    assert torch.allclose(at_half["b"], torch.full((2,), 2.0))
