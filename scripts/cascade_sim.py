#!/usr/bin/env python3
"""E57: shortlist cascade simulation on T28 cached probabilities (eval-only).

Stage 1 scores the full gallery with a cheap model; stage 2 re-scores only the
union of each query's top-K stage-1 images with a second model. Images outside
the shortlist keep stage-1 order below the shortlisted set.

No test images, no test statistics, no network. K is chosen on val only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_KS: tuple[int, ...] = (50, 100, 200)
CODABENCH_GALLERY = 29248
CODABENCH_QUERIES = 893
TIMING_SCALE = 1.275  # same smoke→Codabench factor used for E48/E24


@dataclass(frozen=True)
class TimingModel:
    """Per-image seconds measured on local 35k smoke (gallery-forward dominated)."""

    stage1_name: str
    stage2_name: str
    t1_s_per_image: float
    t2_s_per_image: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="competition data root (annotations under task2/val)",
    )
    parser.add_argument(
        "--stage1-probs",
        type=Path,
        required=True,
        help="npz with key 'data' shape (G, 40); stage-1 (cheap) model",
    )
    parser.add_argument(
        "--stage2-probs",
        type=Path,
        required=True,
        help="npz with key 'data' shape (G, 40); stage-2 model",
    )
    parser.add_argument(
        "--ids",
        type=Path,
        required=True,
        help="JSON list of image paths aligned with both probs rows",
    )
    parser.add_argument(
        "--protocol",
        choices=("id_val",),
        default="id_val",
        help="val protocol (id_val uses queries.csv + eval domains)",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help="shortlist K values",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=20,
        help="number of 893-query draws from val (seed 0)",
    )
    parser.add_argument(
        "--draw-size",
        type=int,
        default=CODABENCH_QUERIES,
        help="queries per draw (Codabench uses 893)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hybrid scorer alpha",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
        help="dual-softmax temperature",
    )
    parser.add_argument(
        "--t1-smoke-s",
        type=float,
        default=704.7,
        help="stage-1 full-gallery smoke seconds on 33407 images (E20 default)",
    )
    parser.add_argument(
        "--t2-smoke-s",
        type=float,
        default=3161.5,
        help="stage-2 full-gallery smoke seconds on 33407 images (E17 default)",
    )
    parser.add_argument(
        "--smoke-gallery",
        type=int,
        default=33407,
        help="gallery size used for smoke timings",
    )
    parser.add_argument(
        "--stage1-name",
        default="e20_dinov2",
    )
    parser.add_argument(
        "--stage2-name",
        default="e17_clip_b16",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs" / "e57_cascade" / "cascade_sim.json",
    )
    return parser.parse_args()


def _load_probs(path: Path) -> np.ndarray:
    loaded = np.load(path)
    if "data" not in loaded.files:
        raise ValueError(f"{path} missing 'data' array; keys={loaded.files}")
    return np.asarray(loaded["data"], dtype=np.float32)


def _score_probs(
    queries: np.ndarray,
    probs: np.ndarray,
    *,
    alpha: float,
    tau: float,
) -> np.ndarray:
    from src.wacv.evaluation import dual_softmax_rerank, ranking_distance

    distances = ranking_distance(
        queries,
        probs,
        distance="hybrid",
        temperature=1.0,
        alpha=alpha,
    )
    return dual_softmax_rerank(distances, tau=tau)


def _true_match_indices(
    query: np.ndarray,
    gallery_attrs: np.ndarray,
) -> np.ndarray:
    matches = np.all(gallery_attrs == query[None, :], axis=1)
    return np.flatnonzero(matches)


def _cascade_distances(
    stage1: np.ndarray,
    stage2: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, int, float]:
    """Build cascade distance matrix; return distances, union size, match recall.

    Shortlisted images use stage-2 distances. Non-shortlisted images keep
    stage-1 distances but are offset so every shortlisted image ranks above
    every non-shortlisted one (stable within each set).
    """
    num_queries, num_gallery = stage1.shape
    shortlist_mask = np.zeros((num_queries, num_gallery), dtype=bool)
    for query_index in range(num_queries):
        top_k = np.argpartition(stage1[query_index], kth=min(k, num_gallery) - 1)[:k]
        shortlist_mask[query_index, top_k] = True
    union_indices = np.flatnonzero(shortlist_mask.any(axis=0))
    union_size = int(union_indices.size)

    # Offset keeps non-shortlist below shortlist while preserving relative order.
    offset = float(np.max(stage1) - np.min(stage1) + 1.0)
    cascaded = stage1.copy()
    cascaded[shortlist_mask] = stage2[shortlist_mask]
    cascaded[~shortlist_mask] = stage1[~shortlist_mask] + offset
    return cascaded, union_size, float("nan")


def _shortlist_recall(
    stage1: np.ndarray,
    queries: np.ndarray,
    gallery_attrs: np.ndarray,
    *,
    k: int,
) -> float:
    recalls: list[float] = []
    num_gallery = stage1.shape[1]
    k_eff = min(k, num_gallery)
    for query_index, query in enumerate(queries):
        match_indices = _true_match_indices(query, gallery_attrs)
        if match_indices.size == 0:
            continue
        top_k = np.argpartition(stage1[query_index], kth=k_eff - 1)[:k_eff]
        top_set = {int(index) for index in top_k.tolist()}
        hit = sum(1 for index in match_indices.tolist() if int(index) in top_set)
        recalls.append(hit / float(match_indices.size))
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def _metric_vector(metrics: Mapping[str, float]) -> np.ndarray:
    keys = ("mADM", "mAP", "Rank-1", "Rank-5", "Rank-10", "mINP")
    return np.asarray([float(metrics[key]) for key in keys], dtype=np.float64)


def _t_est_seconds(
    timing: TimingModel,
    *,
    union_size: int,
    gallery_n: int = CODABENCH_GALLERY,
) -> float:
    raw = gallery_n * timing.t1_s_per_image + union_size * timing.t2_s_per_image
    return float(TIMING_SCALE * raw)


def _draw_query_indices(
    num_queries: int,
    draw_size: int,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    if draw_size > num_queries:
        raise ValueError(
            f"draw_size {draw_size} exceeds available queries {num_queries}"
        )
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.choice(num_queries, size=draw_size, replace=False)
            for _ in range(n_draws)
        ],
        axis=0,
    )


def main() -> int:  # noqa: PLR0915
    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.data import filter_gt_by_domains, load_gt, load_queries
    from src.wacv.monitoring.metrics import evaluate_ranking

    args = parse_args()
    data_root = args.data_root.resolve()
    val_root = data_root / "annotations" / "task2" / "val"
    ground_truth = load_gt(val_root / "gt.csv")
    attribute_names = attribute_names_from_columns(
        tuple(str(column) for column in ground_truth.columns)
    )
    # Match id_val dump domains (all eval domains in organizer val).
    ground_truth = filter_gt_by_domains(ground_truth, None)
    queries = load_queries(val_root / "queries.csv")
    ids = json.loads(args.ids.read_text(encoding="utf-8"))
    if not isinstance(ids, list):
        raise ValueError(f"--ids must be a JSON list; got {type(ids).__name__}")

    gt_by_image = {
        str(row["image"]): row[list(attribute_names)].to_numpy(dtype=np.int64)
        for _, row in ground_truth.iterrows()
    }
    gallery_attrs = np.stack([gt_by_image[str(image_id)] for image_id in ids], axis=0)
    stage1_probs = _load_probs(args.stage1_probs)
    stage2_probs = _load_probs(args.stage2_probs)
    if stage1_probs.shape != stage2_probs.shape:
        raise ValueError(
            "stage1/stage2 shape mismatch: "
            f"{stage1_probs.shape} vs {stage2_probs.shape}"
        )
    if stage1_probs.shape[0] != len(ids):
        raise ValueError(f"probs rows {stage1_probs.shape[0]} != ids length {len(ids)}")
    if gallery_attrs.shape[0] != len(ids):
        raise ValueError("gallery attribute rows do not match ids")

    timing = TimingModel(
        stage1_name=args.stage1_name,
        stage2_name=args.stage2_name,
        t1_s_per_image=float(args.t1_smoke_s) / float(args.smoke_gallery),
        t2_s_per_image=float(args.t2_smoke_s) / float(args.smoke_gallery),
    )

    print("scoring stage1 / stage2 / ensemble …", flush=True)
    stage1_dist = _score_probs(queries, stage1_probs, alpha=args.alpha, tau=args.tau)
    stage2_dist = _score_probs(queries, stage2_probs, alpha=args.alpha, tau=args.tau)
    ensemble_probs = 0.5 * (stage1_probs + stage2_probs)
    ensemble_dist = _score_probs(
        queries, ensemble_probs, alpha=args.alpha, tau=args.tau
    )

    base_stage1 = evaluate_ranking(queries, gallery_attrs, stage1_dist)
    base_stage2 = evaluate_ranking(queries, gallery_attrs, stage2_dist)
    base_ensemble = evaluate_ranking(queries, gallery_attrs, ensemble_dist)
    full_gain = float(base_ensemble["mADM"] - base_stage1["mADM"])

    draw_indices = _draw_query_indices(
        queries.shape[0],
        args.draw_size,
        args.n_draws,
        args.seed,
    )

    per_k: dict[str, object] = {}
    for k in args.ks:
        draw_rows: list[dict[str, object]] = []
        print(f"K={k}: {args.n_draws} draws …", flush=True)
        for draw_id, query_idx in enumerate(draw_indices):
            q = queries[query_idx]
            s1 = stage1_dist[query_idx]
            s2 = stage2_dist[query_idx]
            cascaded, union_size, _ = _cascade_distances(s1, s2, k=k)
            recall = _shortlist_recall(s1, q, gallery_attrs, k=k)
            metrics = evaluate_ranking(q, gallery_attrs, cascaded)
            # Full-ensemble mADM on the same draw for fraction-of-gain.
            ens = evaluate_ranking(q, gallery_attrs, ensemble_dist[query_idx])
            s1_m = evaluate_ranking(q, gallery_attrs, s1)
            gain = float(ens["mADM"] - s1_m["mADM"])
            frac = (
                float(metrics["mADM"] - s1_m["mADM"]) / gain
                if abs(gain) > 1e-12
                else 0.0
            )
            t_est = _t_est_seconds(timing, union_size=union_size)
            draw_rows.append(
                {
                    "draw": draw_id,
                    "union_size": union_size,
                    "recall": recall,
                    "metrics": {key: float(value) for key, value in metrics.items()},
                    "t_est_s": t_est,
                    "fraction_of_full_gain": frac,
                    "stage1_mADM": float(s1_m["mADM"]),
                    "ensemble_mADM": float(ens["mADM"]),
                }
            )
        recalls = np.asarray([row["recall"] for row in draw_rows], dtype=np.float64)
        fracs = np.asarray(
            [row["fraction_of_full_gain"] for row in draw_rows], dtype=np.float64
        )
        unions = np.asarray([row["union_size"] for row in draw_rows], dtype=np.float64)
        t_ests = np.asarray([row["t_est_s"] for row in draw_rows], dtype=np.float64)
        metric_keys = ("mADM", "mAP", "Rank-1", "Rank-5", "Rank-10", "mINP")
        mean_metrics = {
            key: float(
                np.mean([float(row["metrics"][key]) for row in draw_rows])  # type: ignore[index]
            )
            for key in metric_keys
        }
        summary = {
            "k": k,
            "mean": {
                "union_size": float(unions.mean()),
                "recall": float(recalls.mean()),
                "t_est_s": float(t_ests.mean()),
                "fraction_of_full_gain": float(fracs.mean()),
                "metrics": mean_metrics,
            },
            "std": {
                "union_size": float(unions.std(ddof=0)),
                "recall": float(recalls.std(ddof=0)),
                "t_est_s": float(t_ests.std(ddof=0)),
                "fraction_of_full_gain": float(fracs.std(ddof=0)),
            },
            "h57_recall_ok": bool(recalls.mean() >= 0.98),
            "h57_gain_ok": bool(fracs.mean() >= 0.90),
            "h57_time_ok": bool(t_ests.mean() <= 3200.0),
            "draws": draw_rows,
        }
        summary["h57_pass"] = bool(
            summary["h57_recall_ok"]
            and summary["h57_gain_ok"]
            and summary["h57_time_ok"]
        )
        per_k[str(k)] = summary
        print(
            f"  K={k} recall={recalls.mean():.4f} "
            f"frac_gain={fracs.mean():.4f} "
            f"T_est={t_ests.mean():.1f}s "
            f"union={unions.mean():.1f} "
            f"pass={summary['h57_pass']}",
            flush=True,
        )

    any_pass = any(bool(per_k[str(k)]["h57_pass"]) for k in args.ks)
    payload = {
        "protocol": args.protocol,
        "stage1": args.stage1_name,
        "stage2": args.stage2_name,
        "timing": asdict(timing),
        "n_draws": args.n_draws,
        "draw_size": args.draw_size,
        "seed": args.seed,
        "baselines_full_val": {
            "stage1": {key: float(value) for key, value in base_stage1.items()},
            "stage2": {key: float(value) for key, value in base_stage2.items()},
            "ensemble_50_50": {
                key: float(value) for key, value in base_ensemble.items()
            },
            "full_ensemble_mADM_gain_vs_stage1": full_gain,
        },
        "per_k": per_k,
        "h57_any_k_pass": any_pass,
        "h57_verdict": "promoted" if any_pass else "falsified_or_inconclusive",
        "notes": [
            "Stage2 = second model on shortlist only (no hflip TTA yet; E33 pending).",
            "T_est uses smoke per-image seconds * Codabench gallery + union, *1.275.",
            "Full two-model gain = 50/50 probability ensemble (identity TTA).",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    print(f"H57 any-K pass={any_pass} verdict={payload['h57_verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
