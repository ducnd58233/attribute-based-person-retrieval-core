"""Reproduce E70 gap-decomposition tables from a run's cached probabilities.

Reads ``artifacts/id_val_probs.npz`` (+ ``id_val_ids.json``) and organizer val
``gt.csv`` / ``queries.csv``. Scores with the shared scorer registry and
``evaluate_ranking`` — no second copy of the metric formulas.

Example::

    uv run python scripts/gap-diagnostics.py \\
      --artifacts path/to/run/artifacts \\
      --data data \\
      --out runs/t70_gap/e17_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Run artifacts dir with id_val_probs.npz and id_val_ids.json",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data"),
        help="Dataset root (annotations/task2/val)",
    )
    parser.add_argument(
        "--split",
        default="id_val",
        help="Artifact stem prefix (default id_val → id_val_probs.npz)",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument(
        "--scorer-grid",
        action="store_true",
        help="Also score α/τ / no-DS / block-std variants from E70 §5",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _load_aligned(
    artifacts: Path, data_root: Path, split: str
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.data import load_gt, load_queries

    val = data_root / "annotations" / "task2" / "val"
    gt = load_gt(val / "gt.csv")
    names = attribute_names_from_columns(
        tuple(str(c) for c in gt.columns if c != "image")
    )
    ids_path = artifacts / f"{split}_ids.json"
    probs_path = artifacts / f"{split}_probs.npz"
    image_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if not isinstance(image_ids, list):
        raise TypeError(f"{ids_path} must be a JSON list of image paths")
    probs = np.load(probs_path)["data"].astype(np.float32)
    if len(image_ids) != len(probs):
        raise ValueError(
            f"{len(image_ids)} ids but {len(probs)} probability rows in {probs_path}"
        )
    ordered = gt.set_index("image", drop=False).loc[image_ids]
    if list(ordered["image"]) != image_ids:
        raise ValueError("gt.csv could not be aligned to artifact image ids")
    attrs = ordered[list(names)].to_numpy(dtype=np.int64)
    queries = load_queries(val / "queries.csv")
    return queries, attrs, [str(x) for x in image_ids], probs


def _score(
    queries: np.ndarray,
    attrs: np.ndarray,
    probs: np.ndarray,
    *,
    alpha: float,
    temperature: float,
    tau: float | None,
) -> dict[str, float]:
    from src.wacv.evaluation import dual_softmax_rerank, hybrid_distance
    from src.wacv.monitoring.metrics import evaluate_ranking

    distances = hybrid_distance(queries, probs, alpha=alpha, temperature=temperature)
    if tau is not None:
        distances = dual_softmax_rerank(distances, tau=float(tau))
    return evaluate_ranking(queries, attrs, distances)


def _per_dataset(
    queries: np.ndarray,
    attrs: np.ndarray,
    image_ids: list[str],
    probs: np.ndarray,
    *,
    alpha: float,
    temperature: float,
    tau: float,
) -> dict[str, Any]:
    from src.wacv.data import domain_from_image

    domains = sorted({domain_from_image(path) for path in image_ids})
    report: dict[str, Any] = {}
    for domain in domains:
        mask = np.array(
            [domain_from_image(path) == domain for path in image_ids], dtype=bool
        )
        domain_attrs = attrs[mask]
        domain_probs = probs[mask]
        has_match = np.array(
            [bool(np.any(np.all(domain_attrs == query, axis=1))) for query in queries],
            dtype=bool,
        )
        domain_queries = queries[has_match]
        metrics = _score(
            domain_queries,
            domain_attrs,
            domain_probs,
            alpha=alpha,
            temperature=temperature,
            tau=tau,
        )
        mean_r = float(
            np.mean(
                [
                    float(np.all(domain_attrs == query, axis=1).sum())
                    for query in domain_queries
                ]
            )
        )
        report[domain] = {
            "n_queries": len(domain_queries),
            "n_gallery": int(mask.sum()),
            "mean_R": mean_r,
            "metrics": metrics,
        }
    return report


def _match_counts(queries: np.ndarray, attrs: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(queries), dtype=np.int64)
    for index, query in enumerate(queries):
        counts[index] = int(np.all(attrs == query, axis=1).sum())
    return counts


def _per_r_buckets(
    queries: np.ndarray,
    attrs: np.ndarray,
    probs: np.ndarray,
    *,
    alpha: float,
    temperature: float,
    tau: float,
) -> list[dict[str, Any]]:
    from src.wacv.evaluation import dual_softmax_rerank, hybrid_distance
    from src.wacv.monitoring.metrics import evaluate_ranking

    distances = dual_softmax_rerank(
        hybrid_distance(queries, probs, alpha=alpha, temperature=temperature),
        tau=tau,
    )
    counts = _match_counts(queries, attrs)
    edges = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 10_000)]
    buckets: list[dict[str, Any]] = []
    for low, high in edges:
        mask = (counts >= low) & (counts <= high)
        if not mask.any():
            continue
        metrics = evaluate_ranking(queries[mask], attrs, distances[mask])
        label = f"{low}-{high}" if high < 10_000 else f">= {low}"
        buckets.append(
            {
                "bucket": label,
                "n": int(mask.sum()),
                "mean_R": float(counts[mask].mean()),
                "metrics": metrics,
            }
        )
    return buckets


def _block_standardize(distances: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """Per-query z-score of distances inside each domain (E70 H61a probe).

    Exact scratch-code scaling for the ``block_std_then_ds`` cell was not
    recovered; this definition matches E70's ``ds_then_block_std`` (0.6277)
    and still falsifies H61a (≪ S* 0.7085).
    """
    out = distances.copy()
    for group in np.unique(group_ids):
        columns = group_ids == group
        block = out[:, columns]
        mean = block.mean(axis=1, keepdims=True)
        std = block.std(axis=1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        out[:, columns] = (block - mean) / std
    return out


def _scorer_grid(
    queries: np.ndarray,
    attrs: np.ndarray,
    image_ids: list[str],
    probs: np.ndarray,
) -> dict[str, dict[str, float]]:
    from src.wacv.data import domain_from_image
    from src.wacv.evaluation import dual_softmax_rerank, hybrid_distance
    from src.wacv.monitoring.metrics import evaluate_ranking

    group_ids = np.array([domain_from_image(path) for path in image_ids])
    grid: dict[str, dict[str, float]] = {}
    base = hybrid_distance(queries, probs, alpha=0.5, temperature=1.0)
    grid["hybrid_no_ds"] = evaluate_ranking(queries, attrs, base)
    star = dual_softmax_rerank(base, tau=0.05)
    grid["s_star"] = evaluate_ranking(queries, attrs, star)
    for alpha in (0.0, 0.25, 0.75, 1.0):
        distances = dual_softmax_rerank(
            hybrid_distance(queries, probs, alpha=alpha, temperature=1.0), tau=0.05
        )
        grid[f"alpha_{alpha:g}"] = evaluate_ranking(queries, attrs, distances)
    for tau in (0.02, 0.1, 0.2):
        distances = dual_softmax_rerank(
            hybrid_distance(queries, probs, alpha=0.5, temperature=1.0), tau=tau
        )
        grid[f"tau_{tau:g}"] = evaluate_ranking(queries, attrs, distances)
    block_then_ds = dual_softmax_rerank(_block_standardize(base, group_ids), tau=0.05)
    grid["block_std_then_ds"] = evaluate_ranking(queries, attrs, block_then_ds)
    ds_then_block = _block_standardize(star, group_ids)
    grid["ds_then_block_std"] = evaluate_ranking(queries, attrs, ds_then_block)
    return grid


def main() -> None:
    from src.wacv.utils import configure_default_logging

    configure_default_logging()
    args = parse_args()
    queries, attrs, image_ids, probs = _load_aligned(
        args.artifacts.resolve(), args.data.resolve(), args.split
    )
    combined = _score(
        queries,
        attrs,
        probs,
        alpha=args.alpha,
        temperature=args.temperature,
        tau=args.tau,
    )
    per_dataset = _per_dataset(
        queries,
        attrs,
        image_ids,
        probs,
        alpha=args.alpha,
        temperature=args.temperature,
        tau=args.tau,
    )
    weights = np.array(
        [float(per_dataset[name]["n_queries"]) for name in per_dataset],
        dtype=np.float64,
    )
    madms = np.array(
        [float(per_dataset[name]["metrics"]["mADM"]) for name in per_dataset],
        dtype=np.float64,
    )
    weighted_mean = float((madms * weights).sum() / weights.sum())
    payload: dict[str, Any] = {
        "artifacts": str(args.artifacts.resolve()),
        "split": args.split,
        "ranking": {
            "distance": "hybrid",
            "alpha": args.alpha,
            "temperature": args.temperature,
            "query_norm": "dual_softmax",
            "query_norm_tau": args.tau,
        },
        "combined": {
            "n_queries": len(queries),
            "n_gallery": len(image_ids),
            "metrics": combined,
        },
        "per_dataset": per_dataset,
        "query_weighted_per_dataset_mADM": weighted_mean,
        "combined_gallery_penalty": float(combined["mADM"] - weighted_mean),
        "per_r_buckets": _per_r_buckets(
            queries,
            attrs,
            probs,
            alpha=args.alpha,
            temperature=args.temperature,
            tau=args.tau,
        ),
    }
    if args.scorer_grid:
        payload["scorer_grid"] = _scorer_grid(queries, attrs, image_ids, probs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"combined mADM={combined['mADM']:.4f} "
        f"weighted_per_dataset={weighted_mean:.4f} "
        f"penalty={combined['mADM'] - weighted_mean:+.4f} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
