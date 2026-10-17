"""Audit the organizer train/val data: label structure, domain shift, image stats.

Reads only organizer annotations and images (no test data). Writes
``<out>/data_audit.json`` and ``<out>/data_audit.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Attribute index groups that are mutually exclusive in the UPAR schema.
EXCLUSIVE_GROUPS: dict[str, tuple[int, ...]] = {
    "age": (0, 1, 2),
    "hair_length": (4, 5, 6),
    "upper_color": tuple(range(8, 20)),
    "lower_color": tuple(range(21, 33)),
    "lower_type": (33, 34),
}
IMAGE_STAT_NAMES = (
    "width",
    "height",
    "aspect_h_over_w",
    "brightness",
    "contrast",
    "saturation",
    "sharpness_lapvar",
    "bits_per_pixel",
)
PERCENTILES = (5, 25, 50, 75, 95)
GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "data_audit")
    parser.add_argument(
        "--images-per-domain",
        type=int,
        default=1500,
        help="random images per domain and split for pixel statistics",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def label_stats(ground_truth: pd.DataFrame, attribute_names: list[str]) -> dict:
    """Per-domain prevalence, attribute-vector counts and group exclusivity."""
    from src.wacv.data import domain_from_image

    labels = ground_truth[attribute_names].to_numpy(dtype=np.int64)
    domains = ground_truth["image"].map(domain_from_image).to_numpy()
    report: dict = {"n": len(ground_truth), "domains": {}}
    for domain in sorted(set(domains)):
        domain_labels = labels[domains == domain]
        vectors, images_per_vector = np.unique(
            domain_labels, axis=0, return_counts=True
        )
        groups = {}
        for group, indices in EXCLUSIVE_GROUPS.items():
            positives = domain_labels[:, list(indices)].sum(axis=1)
            groups[group] = {
                "zero": float((positives == 0).mean()),
                "one": float((positives == 1).mean()),
                "multi": float((positives > 1).mean()),
            }
        prevalence = domain_labels.mean(axis=0).round(4).tolist()
        report["domains"][domain] = {
            "n": len(domain_labels),
            "prevalence": dict(zip(attribute_names, prevalence, strict=True)),
            "unique_vectors": len(vectors),
            "images_per_vector_median": float(np.median(images_per_vector)),
            "images_per_vector_mean": float(images_per_vector.mean()),
            "singleton_vector_share": float((images_per_vector == 1).mean()),
            "groups": groups,
            "positives_per_image_mean": float(domain_labels.sum(axis=1).mean()),
        }
    return report


def prevalence_shift(train_labels: dict, attribute_names: list[str]) -> list[dict]:
    """Attributes sorted by how much their prevalence differs across domains."""
    domains = sorted(train_labels["domains"])
    rows = []
    for attribute in attribute_names:
        by_domain = {
            domain: train_labels["domains"][domain]["prevalence"][attribute]
            for domain in domains
        }
        spread = max(by_domain.values()) - min(by_domain.values())
        rows.append({"attr": attribute, **by_domain, "range": spread})
    return sorted(rows, key=lambda row: -row["range"])


def query_overlap(
    train_gt: pd.DataFrame, val_gt: pd.DataFrame, attribute_names: list[str]
) -> dict:
    """How many val attribute vectors also occur in train."""
    train_vectors = {
        tuple(row) for row in train_gt[attribute_names].to_numpy(dtype=np.int64)
    }
    val_rows = [tuple(row) for row in val_gt[attribute_names].to_numpy(dtype=np.int64)]
    val_vectors = set(val_rows)
    shared = val_vectors & train_vectors
    seen_rows = np.fromiter((row in train_vectors for row in val_rows), dtype=bool)
    return {
        "val_unique_vectors": len(val_vectors),
        "val_vectors_seen_in_train": len(shared),
        "val_vector_seen_share": len(shared) / max(len(val_vectors), 1),
        "val_images_with_seen_vector_share": float(seen_rows.mean()),
    }


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian: a simple sharpness score."""
    laplacian = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(laplacian.var())


def single_image_stats(path: Path) -> tuple[float, ...]:
    """One row of IMAGE_STAT_NAMES for one image file."""
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb @ GRAY_WEIGHTS
    brightest, darkest = rgb.max(axis=2), rgb.min(axis=2)
    saturation = np.where(
        brightest > 0, (brightest - darkest) / np.maximum(brightest, 1e-6), 0.0
    )
    sharpness = laplacian_variance(gray) if min(width, height) > 2 else 0.0
    return (
        width,
        height,
        height / width,
        float(gray.mean()),
        float(gray.std()),
        float(saturation.mean()),
        sharpness,
        path.stat().st_size * 8 / (width * height),
    )


def image_stats(paths: list[Path]) -> dict:
    """Percentiles of each image statistic over ``paths``."""
    table = np.array([single_image_stats(path) for path in paths], dtype=np.float64)
    widths, heights = table[:, 0], table[:, 1]
    percentile_keys = [f"p{percentile}" for percentile in PERCENTILES]
    return {
        "n": len(table),
        **{
            name: dict(
                zip(
                    percentile_keys,
                    np.percentile(table[:, column], PERCENTILES).round(4).tolist(),
                    strict=True,
                )
            )
            for column, name in enumerate(IMAGE_STAT_NAMES)
        },
        "short_side_lt_64_share": float((np.minimum(widths, heights) < 64).mean()),
        "height_lt_128_share": float((heights < 128).mean()),
    }


def augmix_ops() -> list[str]:
    """Op names torchvision's default AugMix samples from."""
    from torchvision.transforms import AugMix

    return sorted(AugMix()._augmentation_space(10, (224, 112)))


def markdown_table(
    header: list[str], align: list[str], rows: list[list[str]]
) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(align) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# Data audit", ""]
    for split in ("train", "val"):
        rows = [
            [
                domain,
                str(stats["n"]),
                str(stats["unique_vectors"]),
                f"{stats['images_per_vector_median']:.0f}",
                f"{stats['singleton_vector_share']:.3f}",
                f"{stats['positives_per_image_mean']:.2f}",
            ]
            for domain, stats in report["labels"][split]["domains"].items()
        ]
        header = ["domain", "n", "unique vectors", "img/vector median"]
        header += ["singleton share", "pos/img"]
        lines += [f"## Labels: {split}", ""]
        lines += [*markdown_table(header, ["---"] + ["---:"] * 5, rows), ""]

    group_rows = [
        [domain, group, *(f"{shares[key]:.3f}" for key in ("zero", "one", "multi"))]
        for domain, stats in report["labels"]["train"]["domains"].items()
        for group, shares in stats["groups"].items()
    ]
    lines += [
        "## Group exclusivity (train; share of images with 0 / 1 / >1 positives)",
        "",
    ]
    lines += markdown_table(
        ["domain", "group", "zero", "one", "multi"],
        ["---", "---", "---:", "---:", "---:"],
        group_rows,
    )

    domains = sorted(report["labels"]["train"]["domains"])
    shift_rows = [
        [
            row["attr"],
            *(f"{row[domain]:.3f}" for domain in domains),
            f"{row['range']:.3f}",
        ]
        for row in report["prevalence_shift"][:15]
    ]
    lines += ["", "## Largest train prevalence shifts across domains", ""]
    lines += markdown_table(
        ["attr", *domains, "range"],
        ["---"] + ["---:"] * (len(domains) + 1),
        shift_rows,
    )

    image_rows = [
        [
            name,
            *(
                f"{stats[key]['p50']:.3g} [{stats[key]['p5']:.3g}, "
                f"{stats[key]['p95']:.3g}]"
                for key in IMAGE_STAT_NAMES
            ),
            f"{stats['height_lt_128_share']:.3f}",
        ]
        for name, stats in report["images"].items()
    ]
    lines += ["", "## Image statistics (median [p5, p95])", ""]
    lines += markdown_table(
        ["domain", *IMAGE_STAT_NAMES, "h<128"],
        ["---"] + ["---:"] * (len(IMAGE_STAT_NAMES) + 1),
        image_rows,
    )
    lines += [
        "",
        "## Query overlap val vs train",
        "",
        "```json",
        json.dumps(report["query_overlap"], indent=2),
        "```",
        "",
        "## AugMix op space (torchvision default all_ops=True)",
        "",
        ", ".join(report["augmix_ops"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def sampled_image_stats(
    ground_truth: pd.DataFrame,
    data_root: Path,
    images_per_domain: int,
    rng: np.random.Generator,
) -> dict[str, dict]:
    """Image statistics on a random sample of each domain."""
    from src.wacv.data import domain_from_image

    domains = ground_truth["image"].map(domain_from_image)
    stats = {}
    for domain in sorted(domains.unique()):
        domain_images = ground_truth.loc[domains == domain, "image"].to_numpy()
        sample_size = min(images_per_domain, len(domain_images))
        sample = rng.choice(domain_images, size=sample_size, replace=False)
        stats[domain] = image_stats([data_root / str(image) for image in sample])
    return stats


def main() -> None:
    from src.wacv.data import load_gt

    args = parse_args()
    annotations = args.data_root / "annotations" / "task2"
    train_gt = load_gt(annotations / "train" / "gt.csv")
    val_gt = load_gt(annotations / "val" / "gt.csv")
    attribute_names = [column for column in train_gt.columns if column != "image"]

    rng = np.random.default_rng(args.seed)
    images = {}
    for split, ground_truth in (("train", train_gt), ("val", val_gt)):
        per_domain = sampled_image_stats(
            ground_truth, args.data_root, args.images_per_domain, rng
        )
        images.update(
            {f"{split}/{domain}": stats for domain, stats in per_domain.items()}
        )

    train_labels = label_stats(train_gt, attribute_names)
    report = {
        "labels": {
            "train": train_labels,
            "val": label_stats(val_gt, attribute_names),
        },
        "prevalence_shift": prevalence_shift(train_labels, attribute_names),
        "query_overlap": query_overlap(train_gt, val_gt, attribute_names),
        "images": images,
        "augmix_ops": augmix_ops(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report, args.out / "data_audit.md")
    print(f"wrote {args.out / 'data_audit.md'}")


if __name__ == "__main__":
    main()
