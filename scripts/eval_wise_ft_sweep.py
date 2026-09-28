"""Evaluate pre-built WiSE-FT checkpoints (E18)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from scripts.evaluate import evaluate_checkpoint
    from src.wacv.utils import configure_default_logging

    configure_default_logging()
    config = ROOT / "configs" / "e18_wise_ft.yml"
    out_dir = ROOT / "runs" / "e18_wise_ft" / "checkpoints"
    run_dir = ROOT / "runs" / "e18_wise_ft"
    alphas = [0.3, 0.5, 0.7]
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    best_alpha: float | None = None
    best_madm = float("-inf")
    for alpha in alphas:
        stem = f"wise_ft_a{alpha:.2f}".replace(".", "")
        checkpoint = out_dir / f"{stem}.pt"
        print(f"evaluating alpha={alpha} {checkpoint}", flush=True)
        results, _ = evaluate_checkpoint(
            config, checkpoint, protocols=("id_val", "lodo_m")
        )
        all_metrics[f"alpha_{alpha:.2f}"] = results
        id_madm = float(results["id_val"]["madm"])
        print(
            f"alpha={alpha:.2f} id_val mADM={id_madm:.4f} "
            f"lodo={results['lodo_m']['madm']:.4f}",
            flush=True,
        )
        if id_madm > best_madm:
            best_madm = id_madm
            best_alpha = alpha
    assert best_alpha is not None
    best_stem = f"wise_ft_a{best_alpha:.2f}".replace(".", "")
    (out_dir / "best.pt").write_bytes((out_dir / f"{best_stem}.pt").read_bytes())
    chosen = all_metrics[f"alpha_{best_alpha:.2f}"]
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "id_val": chosen["id_val"],
                "lodo_m": chosen["lodo_m"],
                "wise_ft_alpha": best_alpha,
                "wise_ft_sweep": all_metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"selected alpha={best_alpha:.2f} id_val mADM={best_madm:.4f}", flush=True)


if __name__ == "__main__":
    main()
