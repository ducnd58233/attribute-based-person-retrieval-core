"""Build a uniform or greedy model soup from compatible checkpoints (E22 / H22).

Example (after >= 3 full-data runs of the same architecture / init):

  uv run python scripts/soup.py \\
    --mode greedy \\
    --checkpoints runs/seed0/checkpoints/best.pt runs/seed1/checkpoints/best.pt \\
                  runs/seed2/checkpoints/best.pt \\
    --metrics runs/e22_model_soup/ingredient_metrics.json \\
    --config configs/e22_model_soup.yml \\
    --out runs/e22_model_soup/checkpoints/soup.pt

``ingredient_metrics.json`` maps checkpoint path -> ID-val mADM (greedy order).
Greedy mode needs ``--config`` so candidate soups can be scored on ID-val;
uniform mode only needs the checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SoupModeName = Literal["uniform", "greedy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("uniform", "greedy"),
        required=True,
        help="uniform averages all; greedy adds by ID-val score (needs --config)",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="compatible .pt checkpoints sharing architecture and initialisation",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="JSON map path->mADM (or list of {checkpoint, madm}) for greedy order",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="experiment YAML used to score greedy candidates on ID-val mADM",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output soup checkpoint path",
    )
    parser.add_argument(
        "--ingredients-out",
        type=Path,
        default=None,
        help="defaults to <out-dir>/ingredients.json",
    )
    return parser.parse_args()


def _checkpoint_key(path: Path) -> str:
    return str(path)


def _metrics_from_list(payload: list[object]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            raise TypeError(
                f"metrics list entries must be objects, got {type(entry)!r}"
            )
        raw_path = entry.get("checkpoint", entry.get("path"))
        raw_metric = entry.get("madm", entry.get("metric", entry.get("mADM")))
        if raw_path is None or raw_metric is None:
            raise KeyError("metrics list entries need checkpoint/path and madm/metric")
        resolved[str(raw_path)] = float(raw_metric)
    return resolved


def _parse_metrics_payload(payload: object) -> dict[str, float]:
    if isinstance(payload, dict):
        return {str(key): float(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return _metrics_from_list(payload)
    raise TypeError(
        f"metrics JSON must be an object or list, got {type(payload).__name__}"
    )


def _lookup_metric(
    checkpoint: Path,
    resolved: Mapping[str, float],
    by_name: Mapping[str, float],
    by_resolved: Mapping[str, float],
) -> float | None:
    key = _checkpoint_key(checkpoint)
    if key in resolved:
        return resolved[key]
    resolved_key = str(checkpoint.resolve())
    if resolved_key in by_resolved:
        return by_resolved[resolved_key]
    if checkpoint.name in by_name:
        return by_name[checkpoint.name]
    return None


def load_ingredient_metrics(
    metrics_path: Path | None,
    checkpoints: Sequence[Path],
) -> dict[str, float]:
    """Parse optional metrics JSON into checkpoint-path -> score."""
    if metrics_path is None:
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    resolved = _parse_metrics_payload(payload)
    by_name = {Path(key).name: value for key, value in resolved.items()}
    by_resolved = {str(Path(key).resolve()): value for key, value in resolved.items()}
    matched: dict[str, float] = {}
    for checkpoint in checkpoints:
        value = _lookup_metric(checkpoint, resolved, by_name, by_resolved)
        if value is not None:
            matched[_checkpoint_key(checkpoint)] = value
    return matched


def _build_id_val_score_fn(
    config_path: Path,
) -> Callable[[Mapping[str, torch.Tensor]], float]:
    """Score an in-memory state dict by ID-val mADM (lazy imports)."""
    from scripts.evaluate import _evaluate_protocol, _prepare_protocol_data
    from src.wacv.config import load_config
    from src.wacv.engine import resolve_device
    from src.wacv.models import build_model

    config = load_config(config_path)
    device = resolve_device(config.run.device)
    ground_truth, queries, attribute_names = _prepare_protocol_data(config, "id_val")
    inference_model = config.model.model_copy(update={"pretrained": False})
    model = build_model(inference_model, num_attributes=len(attribute_names)).to(device)

    def score_fn(state_dict: Mapping[str, torch.Tensor]) -> float:
        model.load_state_dict(dict(state_dict), strict=True)
        metrics = _evaluate_protocol(
            config=config,
            model=model,
            device=device,
            ground_truth=ground_truth,
            queries=queries,
            attribute_names=attribute_names,
            distance=config.ranking.distance,
        )
        return float(metrics["mADM"])

    return score_fn


def write_ingredients_manifest(
    path: Path,
    *,
    mode: SoupModeName,
    checkpoints: Sequence[Path],
    metrics: Mapping[str, float],
    selected: Sequence[str],
) -> None:
    selected_set = set(selected)
    considered = []
    for checkpoint in checkpoints:
        key = _checkpoint_key(checkpoint)
        considered.append(
            {
                "checkpoint": key,
                "metric": metrics.get(key),
                "selected": key in selected_set or checkpoint.name in selected_set,
            }
        )
    payload = {
        "mode": mode,
        "selected": list(selected),
        "considered": considered,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    from src.wacv.soup import (
        SoupIngredient,
        extract_model_state,
        greedy_soup,
        uniform_soup,
        write_soup_checkpoint,
    )

    args = parse_args()
    mode = cast(SoupModeName, args.mode)
    checkpoints = [Path(path) for path in args.checkpoints]
    if len(checkpoints) < 1:
        raise SystemExit("at least one --checkpoints path is required")
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint not found: {checkpoint}")

    metrics = load_ingredient_metrics(args.metrics, checkpoints)
    ingredients = [
        SoupIngredient(
            name=_checkpoint_key(checkpoint),
            state_dict=extract_model_state(checkpoint),
            metric=metrics.get(_checkpoint_key(checkpoint)),
        )
        for checkpoint in checkpoints
    ]

    if mode == "uniform":
        result = uniform_soup(ingredients)
    else:
        if args.config is None:
            raise SystemExit(
                "greedy mode requires --config so candidate soups can be "
                "scored on ID-val mADM (or use --mode uniform)"
            )
        if len(ingredients) < 2:
            raise SystemExit("greedy mode expects at least 2 checkpoints")
        score_fn = _build_id_val_score_fn(Path(args.config))
        result = greedy_soup(ingredients, score_fn)

    out_path = Path(args.out)
    write_soup_checkpoint(result, out_path)
    ingredients_out = args.ingredients_out or (out_path.parent / "ingredients.json")
    write_ingredients_manifest(
        Path(ingredients_out),
        mode=mode,
        checkpoints=checkpoints,
        metrics=metrics,
        selected=result.selected,
    )
    print(f"wrote soup ({result.mode}) -> {out_path}")
    print(f"selected: {list(result.selected)}")
    print(f"ingredients manifest -> {ingredients_out}")


if __name__ == "__main__":
    main()
