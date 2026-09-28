"""Unit tests for greedy / uniform model soup helpers (E22)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.soup import load_ingredient_metrics
from src.wacv.soup import (
    SoupIngredient,
    average_state_dicts,
    greedy_soup,
    order_ingredients,
    uniform_soup,
    write_soup_checkpoint,
)


def _ingredient(
    name: str,
    weight: float,
    *,
    metric: float | None = None,
) -> SoupIngredient:
    return SoupIngredient(
        name=name,
        state_dict={"w": torch.full((2, 2), weight)},
        metric=metric,
    )


def test_average_state_dicts_uniform_mean() -> None:
    a = {"w": torch.zeros(2, 2), "b": torch.ones(2)}
    b = {"w": torch.ones(2, 2), "b": torch.full((2,), 3.0)}
    merged = average_state_dicts([a, b])
    assert torch.allclose(merged["w"], torch.full((2, 2), 0.5))
    assert torch.allclose(merged["b"], torch.full((2,), 2.0))


def test_average_state_dicts_keeps_non_floating_from_first() -> None:
    a = {"idx": torch.tensor([1, 2], dtype=torch.int64)}
    b = {"idx": torch.tensor([9, 8], dtype=torch.int64)}
    merged = average_state_dicts([a, b])
    assert merged["idx"].dtype == torch.int64
    assert torch.equal(merged["idx"], a["idx"])


def test_order_ingredients_by_metric_descending() -> None:
    ingredients = [
        _ingredient("low", 0.0, metric=0.1),
        _ingredient("high", 1.0, metric=0.9),
        _ingredient("mid", 0.5, metric=0.5),
        _ingredient("unknown", 0.2, metric=None),
    ]
    ordered = order_ingredients(ingredients)
    assert [item.name for item in ordered] == ["high", "mid", "low", "unknown"]


def test_uniform_soup_selects_all() -> None:
    ingredients = [
        _ingredient("a", 0.0, metric=0.4),
        _ingredient("b", 2.0, metric=0.6),
    ]
    result = uniform_soup(ingredients)
    assert result.mode == "uniform"
    assert result.selected == ("a", "b")
    assert torch.allclose(result.state_dict["w"], torch.ones(2, 2))


def test_greedy_soup_rejects_when_average_hurts() -> None:
    ingredients = [
        _ingredient("a", 0.0, metric=0.2),
        _ingredient("b", 1.0, metric=0.9),
        _ingredient("c", 0.5, metric=0.5),
    ]
    target = torch.ones(2, 2)

    def score_fn(state: dict[str, torch.Tensor]) -> float:
        return -float(torch.mean((state["w"] - target) ** 2).item())

    result = greedy_soup(ingredients, score_fn)
    assert result.mode == "greedy"
    # Best alone is exact match; averaging with others increases MSE.
    assert result.selected == ("b",)


def test_greedy_soup_keeps_improving_members() -> None:
    ingredients = [
        _ingredient("far", 0.0, metric=0.1),
        _ingredient("near", 0.8, metric=0.8),
        _ingredient("closer", 1.0, metric=0.7),
    ]
    target = torch.ones(2, 2)

    def score_fn(state: dict[str, torch.Tensor]) -> float:
        return -float(torch.mean((state["w"] - target) ** 2).item())

    result = greedy_soup(ingredients, score_fn)
    # Order: near (0.8), closer (0.7), far (0.1).
    # Start near (0.8). Add closer -> avg 0.9, better. Add far -> worse.
    assert result.selected == ("near", "closer")
    assert torch.allclose(result.state_dict["w"], torch.full((2, 2), 0.9))


def test_write_soup_checkpoint_roundtrip(tmp_path: Path) -> None:
    ingredients = [_ingredient("only", 1.0, metric=0.5)]
    result = uniform_soup(ingredients)
    path = tmp_path / "soup.pt"
    write_soup_checkpoint(result, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "model" in payload
    assert payload["soup"]["mode"] == "uniform"
    assert payload["soup"]["selected"] == ["only"]
    assert torch.allclose(payload["model"]["w"], torch.ones(2, 2))


def test_load_ingredient_metrics_map_and_list(tmp_path: Path) -> None:
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.write_bytes(b"")
    b.write_bytes(b"")
    map_path = tmp_path / "metrics_map.json"
    map_path.write_text(json.dumps({str(a): 0.5, "b.pt": 0.4}), encoding="utf-8")
    mapped = load_ingredient_metrics(map_path, [a, b])
    assert mapped[str(a)] == 0.5
    assert mapped[str(b)] == 0.4

    list_path = tmp_path / "metrics_list.json"
    list_path.write_text(
        json.dumps(
            [
                {"checkpoint": str(a), "madm": 0.55},
                {"path": str(b), "metric": 0.45},
            ]
        ),
        encoding="utf-8",
    )
    listed = load_ingredient_metrics(list_path, [a, b])
    assert listed[str(a)] == 0.55
    assert listed[str(b)] == 0.45
