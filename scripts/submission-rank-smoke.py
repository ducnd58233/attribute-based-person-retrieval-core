"""Model-agnostic Codabench-shaped rank_gallery smoke for any packaged submission.

Runs inside ``deployments/docker`` (submission at ``/app``, competition-style
data at ``/data``) or locally with ``--submission`` / ``--data``. Works for
ConvNeXt, CLIP TorchScript, LoRA packages, etc. — only needs a valid
``run.py`` + ``assets/`` layout and a val ``gt.csv`` with image paths.

Examples (from repo root, conda env ``wacv`` active)::

    docker compose -f deployments/docker/docker-compose.yml run --rm \\
      -e HARNESS_N_GALLERY=2048 -e HARNESS_N_QUERIES=512 \\
      submission-rank-smoke

    uv run python scripts/submission-rank-smoke.py \\
      --submission build/submission \\
      --data data \\
      --n-gallery 512 --n-queries 128
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# resource is Linux/macOS only (Docker eval image); optional on Windows hosts.
try:
    import resource as _resource_mod
except ImportError:  # pragma: no cover - Windows local path
    _resource_mod = None

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path(os.environ.get("HARNESS_SUBMISSION", "/app")),
        help="Packaged submission folder (contains run.py)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("HARNESS_DATA", "/data")),
        help="Dataset root (annotations/ + image trees)",
    )
    parser.add_argument(
        "--gt-csv",
        type=Path,
        default=None,
        help="Gallery CSV (default: <data>/annotations/task1/val/gt.csv)",
    )
    parser.add_argument(
        "--n-gallery",
        type=int,
        default=int(os.environ.get("HARNESS_N_GALLERY", "33407")),
        help="Gallery size cap (Codabench-shaped default 33407)",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=int(os.environ.get("HARNESS_N_QUERIES", "3462")),
        help="Query count (Codabench-shaped default 3462)",
    )
    return parser.parse_args()


def _peak_rss_mb() -> float:
    if _resource_mod is None:
        return float("nan")
    return _resource_mod.getrusage(_resource_mod.RUSAGE_SELF).ru_maxrss / 1024.0


def _resolve_image(data_root: Path, rel: str) -> Path:
    rest = rel[len("Market1501/") :] if rel.startswith("Market1501/") else rel
    for candidate in (
        data_root / "Market1501" / rest,
        data_root / "Market-1501-v15.09.15" / rest,
        data_root / rel,
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve gallery image: {rel}")


def build_sample(
    *,
    gt_csv: Path,
    data_root: Path,
    n_gallery: int,
    n_queries: int,
) -> dict[str, Any]:
    with gt_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {gt_csv}")
    img_key = next(k for k in rows[0] if "image" in k.lower())
    attr_cols = [c for c in rows[0] if c != img_key]
    gallery: list[dict[str, Any]] = []
    attr_matrix: list[list[float]] = []
    for index, row in enumerate(rows[: max(n_gallery, 0)]):
        rel = row[img_key]
        path = _resolve_image(data_root, rel)
        gallery.append({"index": index, "image": rel, "image_path": str(path)})
        attr_matrix.append([float(row[c]) for c in attr_cols])
    attrs = np.asarray(attr_matrix, dtype=np.float32)
    if len(attrs) >= n_queries:
        queries = attrs[:n_queries]
    else:
        reps = (n_queries + max(len(attrs), 1) - 1) // max(len(attrs), 1)
        queries = np.tile(attrs, (reps, 1))[:n_queries]
    return {
        "attribute_names": attr_cols,
        "queries": queries.tolist(),
        "gallery": gallery,
    }


def main() -> int:
    args = parse_args()
    submission = args.submission.resolve()
    data_root = args.data.resolve()
    gt_csv = (
        args.gt_csv.resolve()
        if args.gt_csv is not None
        else (data_root / "annotations" / "task1" / "val" / "gt.csv")
    )
    if not (submission / "run.py").is_file():
        raise FileNotFoundError(f"No run.py under {submission}")
    if not gt_csv.is_file():
        raise FileNotFoundError(gt_csv)

    sys.path.insert(0, str(submission))
    print(f"[rank-smoke] peak_rss_start_mb={_peak_rss_mb():.1f}", flush=True)
    t0 = time.perf_counter()
    sample = build_sample(
        gt_csv=gt_csv,
        data_root=data_root,
        n_gallery=args.n_gallery,
        n_queries=args.n_queries,
    )
    print(
        f"[rank-smoke] sample Q={len(sample['queries'])} "
        f"G={len(sample['gallery'])} "
        f"build_s={time.perf_counter() - t0:.1f} "
        f"peak_rss_mb={_peak_rss_mb():.1f}",
        flush=True,
    )
    import run

    t1 = time.perf_counter()
    run.load_model()
    device = getattr(run, "_STATE", {}).get("device")
    print(
        f"[rank-smoke] load_model_s={time.perf_counter() - t1:.1f} "
        f"device={device} peak_rss_mb={_peak_rss_mb():.1f}",
        flush=True,
    )
    t2 = time.perf_counter()
    try:
        result = run.rank_gallery(sample)
    except Exception:
        print("[rank-smoke] rank_gallery FAILED", flush=True)
        traceback.print_exc()
        print(f"[rank-smoke] peak_rss_mb={_peak_rss_mb():.1f}", flush=True)
        return 1
    elapsed = time.perf_counter() - t2
    distances = result.get("distances")
    shape = getattr(distances, "shape", None)
    print(
        f"[rank-smoke] rank_gallery_ok shape={shape} "
        f"elapsed_s={elapsed:.1f} "
        f"peak_rss_mb={_peak_rss_mb():.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
