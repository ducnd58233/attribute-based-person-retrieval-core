"""E74b: sweep latent kNN smoothing (k, beta, similarity gate) on validation.

Predicts ID-val once (probabilities and pooled embeddings), caches both, then
scores every grid cell per dataset gallery. The cell is chosen on the odd
queries of ``--select-domain`` and reported on the even ones, so the number
that matters was never used for the choice. Organizer validation only; no test
image is touched.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.wacv.config import ExperimentConfig, NeighbourSmoothConfig, RankingConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DISPLAY_METRICS = ("mADM", "mAP", "Rank-1", "mINP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="npz of probabilities and embeddings; reused when it exists",
    )
    parser.add_argument("--out", type=Path, required=True, help="result JSON")
    parser.add_argument(
        "--gallery",
        choices=("per-dataset", "combined"),
        default="per-dataset",
        help="rank each dataset on its own, or pool them as the hidden test does",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--beta", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument(
        "--min-similarity",
        type=float,
        nargs="+",
        default=[-1.0, 0.9, 0.93, 0.95, 0.97],
        help="cosine gates (CLIP's top-5 neighbours sit at 0.89 to 0.98); "
        "-1 disables the gate",
    )
    parser.add_argument("--select-domain", default="Market1501")
    parser.add_argument(
        "--harm-limit",
        type=float,
        default=0.005,
        help="a cell is eligible only if no other group's odd-query mADM drops "
        "by more than this",
    )
    return parser.parse_args()


def _predict_to_cache(
    config: ExperimentConfig, checkpoint: Path, cache: Path
) -> dict[str, np.ndarray]:
    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.checkpoint import load_model_weights
    from src.wacv.data import (
        build_eval_loader,
        filter_gt_by_domains,
        load_gt,
        load_queries,
    )
    from src.wacv.engine import resolve_device
    from src.wacv.evaluation import predict_pooled_embeddings, predict_probabilities
    from src.wacv.models import build_model

    val_root = config.data.root / "annotations" / "task2" / "val"
    ground_truth = filter_gt_by_domains(
        load_gt(val_root / "gt.csv"), config.data.eval_domains
    )
    names = attribute_names_from_columns(tuple(str(c) for c in ground_truth.columns))
    device = resolve_device(config.run.device)
    model_config = config.model.model_copy(update={"pretrained": False})
    model = build_model(model_config, num_attributes=len(names)).to(device)
    load_model_weights(checkpoint.resolve(), model, map_location=device)
    loader = build_eval_loader(
        ground_truth, names, config.model.name, config.data, device
    )
    progress = config.run.progress
    arrays = {
        "probabilities": predict_probabilities(
            model,
            loader,
            device,
            progress=progress,
            tta=config.inference.tta,
            tta_scale=float(config.inference.tta_scale),
        ),
        "embeddings": predict_pooled_embeddings(
            model, loader, device, progress=progress
        ),
        "images": ground_truth["image"].to_numpy(dtype=str),
        "gallery_attrs": ground_truth[list(names)].to_numpy(dtype=np.int64),
        "queries": load_queries(val_root / "queries.csv"),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **arrays)
    return arrays


def _groups(images: np.ndarray, gallery: str) -> dict[str, np.ndarray]:
    from src.wacv.data import domain_from_image

    if gallery == "combined":
        return {"combined": np.ones(len(images), dtype=bool)}
    domains = np.array([domain_from_image(str(image)) for image in images])
    return {name: domains == name for name in sorted(set(domains.tolist()))}


def _queries_present(queries: np.ndarray, gallery_attrs: np.ndarray) -> np.ndarray:
    present = {tuple(row) for row in gallery_attrs.tolist()}
    keep = [tuple(row) in present for row in queries.tolist()]
    return queries[np.array(keep, dtype=bool)]


def _score(
    queries: np.ndarray,
    gallery_attrs: np.ndarray,
    probabilities: np.ndarray,
    ranking: RankingConfig,
) -> dict[str, dict[str, float]]:
    from src.wacv.evaluation import rank_distances
    from src.wacv.monitoring.metrics import evaluate_ranking

    distances = rank_distances(
        queries, probabilities, ranking, temperature=float(ranking.temperature)
    )
    odd = np.arange(len(queries)) % 2 == 1
    result: dict[str, dict[str, float]] = {}
    for split, mask in (("all", np.ones_like(odd)), ("odd", odd), ("even", ~odd)):
        metrics = evaluate_ranking(queries[mask], gallery_attrs, distances[mask])
        result[split] = {key: float(metrics[key]) for key in DISPLAY_METRICS}
    return result


def _cell_name(smooth: NeighbourSmoothConfig) -> str:
    return f"k{smooth.k}_b{smooth.beta}_g{smooth.min_similarity}"


def _select_cell(
    results: dict[str, dict[str, dict[str, dict[str, float]]]],
    baseline: dict[str, dict[str, dict[str, float]]],
    select_domain: str,
    harm_limit: float,
) -> str | None:
    """Best odd-query mADM gain on ``select_domain`` among cells with little harm."""
    best: tuple[float, str] | None = None
    for cell, groups in results.items():
        deltas = {
            name: groups[name]["odd"]["mADM"] - baseline[name]["odd"]["mADM"]
            for name in groups
        }
        harm = min(
            (delta for name, delta in deltas.items() if name != select_domain),
            default=0.0,
        )
        if harm < -harm_limit:
            continue
        gain = deltas[select_domain]
        if best is None or gain > best[0]:
            best = (gain, cell)
    return None if best is None else best[1]


def main() -> None:
    from src.wacv.config import NeighbourSmoothConfig, load_config
    from src.wacv.evaluation import neighbour_smooth_probabilities
    from src.wacv.utils import configure_default_logging

    configure_default_logging()
    args = parse_args()
    config = load_config(args.config)
    cached = (
        dict(np.load(args.cache))
        if args.cache.is_file()
        else _predict_to_cache(config, args.checkpoint, args.cache)
    )
    ranking = config.ranking.model_copy(update={"neighbour_smooth": None})
    groups = _groups(cached["images"], args.gallery)
    if args.select_domain not in groups:
        raise SystemExit(
            f"--select-domain {args.select_domain!r} not in groups {sorted(groups)}"
        )

    prepared = {}
    for name, mask in groups.items():
        attrs = cached["gallery_attrs"][mask]
        prepared[name] = {
            "attrs": attrs,
            "probabilities": cached["probabilities"][mask],
            "embeddings": cached["embeddings"][mask],
            "queries": _queries_present(cached["queries"], attrs),
        }

    baseline = {
        name: _score(g["queries"], g["attrs"], g["probabilities"], ranking)
        for name, g in prepared.items()
    }
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for k, beta, gate in itertools.product(args.k, args.beta, args.min_similarity):
        smooth = NeighbourSmoothConfig(k=k, beta=beta, min_similarity=gate)
        results[_cell_name(smooth)] = {
            name: _score(
                g["queries"],
                g["attrs"],
                neighbour_smooth_probabilities(
                    g["probabilities"], g["embeddings"], smooth
                ),
                ranking,
            )
            for name, g in prepared.items()
        }

    selected = _select_cell(results, baseline, args.select_domain, args.harm_limit)
    payload = {
        "gallery": args.gallery,
        "select_domain": args.select_domain,
        "harm_limit": args.harm_limit,
        "baseline": baseline,
        "grid": results,
        "selected_on_odd": selected,
        "selected_even_report": None if selected is None else results[selected],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"selected cell on odd {args.select_domain} queries: {selected}")
    if selected is not None:
        for name in prepared:
            before = baseline[name]["even"]["mADM"]
            after = results[selected][name]["even"]["mADM"]
            delta = after - before
            print(f"  {name} even mADM {before:.4f} -> {after:.4f} ({delta:+.4f})")


if __name__ == "__main__":
    main()
