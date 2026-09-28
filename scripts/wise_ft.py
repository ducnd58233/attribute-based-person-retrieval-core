"""Build WiSE-FT checkpoints from an E17 CLIP fine-tune (H18 / E18)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "e18_wise_ft.yml",
        help="Experiment config (supplies ranking / inference / data paths)",
    )
    parser.add_argument(
        "--fine-tuned",
        type=Path,
        default=ROOT / "runs" / "e17_clip_b16_lpft" / "checkpoints" / "best.pt",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="defaults to runs/<run.name>/checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    from src.wacv.attributes import attribute_names_from_gt
    from src.wacv.config import load_config
    from src.wacv.utils import configure_default_logging
    from src.wacv.wise_ft import write_wise_ft_checkpoint

    configure_default_logging()
    args = parse_args()
    config = load_config(args.config)
    gt_path = Path(config.data.root) / "annotations" / "task2" / "val" / "gt.csv"
    attribute_names = attribute_names_from_gt(gt_path)
    out_dir = args.out_dir or (
        Path(config.run.output_dir) / config.run.name / "checkpoints"
    )
    out_dir = Path(out_dir)
    written: dict[str, str] = {}
    for alpha in args.alphas:
        stem = f"wise_ft_a{alpha:.2f}".replace(".", "")
        path = out_dir / f"{stem}.pt"
        write_wise_ft_checkpoint(
            fine_tuned_checkpoint=args.fine_tuned,
            attribute_names=attribute_names,
            alpha=alpha,
            output_path=path,
            dropout=float(config.model.dropout),
        )
        written[f"{alpha:.2f}"] = str(path)
        print(f"wrote alpha={alpha:.2f} -> {path}")

    manifest = out_dir / "wise_ft_manifest.json"
    manifest.write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
