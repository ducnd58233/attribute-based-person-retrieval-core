"""Visualize first-N images: ground-truth attributes vs model predictions."""

from __future__ import annotations

import argparse
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
        default=ROOT / "configs" / "convnext_baseline.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="defaults to runs/<run-name>/checkpoints/best.pt",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="task1 split whose first rows are shown",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="number of leading samples to score and plot",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="probability threshold for a positive attribute",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="optional path to save the comparison figure",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="skip interactive plt.show (still saves when --out is set)",
    )
    return parser.parse_args()


def main() -> None:
    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.checkpoint import load_model_weights
    from src.wacv.config import load_config
    from src.wacv.data import AttributeDataset, load_gt
    from src.wacv.engine import resolve_device
    from src.wacv.models import build_model
    from src.wacv.visualize import (
        collect_comparisons,
        plot_comparisons,
        print_comparison_summary,
    )

    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config.run.device)
    checkpoint = args.checkpoint or (
        config.run.output_dir / config.run.name / "checkpoints" / "best.pt"
    )
    if not checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {checkpoint}")

    gt_path = config.data.root / "annotations" / "task1" / args.split / "gt.csv"
    ground_truth = load_gt(gt_path)
    attribute_names = attribute_names_from_columns(
        tuple(str(column) for column in ground_truth.columns)
    )
    dataset = AttributeDataset(
        ground_truth, config.data.root, attribute_names=attribute_names
    )
    image_paths = [str(path) for path in ground_truth["image"].tolist()]

    inference_model = config.model.model_copy(update={"pretrained": False})
    model = build_model(inference_model, num_attributes=len(attribute_names)).to(device)
    load_model_weights(checkpoint.resolve(), model, map_location=device)

    comparisons = collect_comparisons(
        model,
        dataset,
        image_paths,
        device,
        limit=args.limit,
        threshold=args.threshold,
    )
    print_comparison_summary(comparisons)
    plot_comparisons(
        comparisons,
        title=(
            f"Label vs prediction — task1/{args.split} "
            f"first {len(comparisons)} (ckpt={checkpoint.name})"
        ),
        save_path=args.out,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
