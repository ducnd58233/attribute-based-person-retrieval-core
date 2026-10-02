"""Compare ground-truth attribute labels against model predictions on images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.wacv.data import AttributeDataset
from src.wacv.normalization import NormalizationStats


@dataclass(frozen=True)
class SampleComparison:
    """One image with positive GT attributes vs predicted positives."""

    image_relpath: str
    rgb: np.ndarray
    label_positives: tuple[str, ...]
    pred_positives: tuple[str, ...]
    true_positives: tuple[str, ...]
    false_positives: tuple[str, ...]
    false_negatives: tuple[str, ...]


def tensor_to_display_rgb(image: torch.Tensor, stats: NormalizationStats) -> np.ndarray:
    """Undo ``stats`` normalisation; return an HxWx3 float array in [0, 1]."""
    mean = np.asarray(stats.mean, dtype=np.float32)[:, None, None]
    std = np.asarray(stats.std, dtype=np.float32)[:, None, None]
    rgb = (image.detach().cpu().numpy() * std + mean).clip(0.0, 1.0)
    return np.transpose(rgb, (1, 2, 0))


def positive_names(
    flags: np.ndarray, attribute_names: tuple[str, ...]
) -> tuple[str, ...]:
    """Map a binary attribute vector to the names that are on."""
    return tuple(
        name for name, flag in zip(attribute_names, flags, strict=True) if bool(flag)
    )


def compare_sample(
    *,
    image_relpath: str,
    image: torch.Tensor,
    targets: np.ndarray,
    probabilities: np.ndarray,
    attribute_names: tuple[str, ...],
    stats: NormalizationStats,
    threshold: float = 0.5,
) -> SampleComparison:
    """Build a label-vs-prediction comparison for a single sample."""
    predicted = probabilities >= threshold
    label_set = set(positive_names(targets.astype(bool), attribute_names))
    pred_set = set(positive_names(predicted, attribute_names))
    return SampleComparison(
        image_relpath=image_relpath,
        rgb=tensor_to_display_rgb(image, stats),
        label_positives=tuple(sorted(label_set)),
        pred_positives=tuple(sorted(pred_set)),
        true_positives=tuple(sorted(label_set & pred_set)),
        false_positives=tuple(sorted(pred_set - label_set)),
        false_negatives=tuple(sorted(label_set - pred_set)),
    )


def collect_comparisons(
    model: nn.Module,
    dataset: AttributeDataset,
    gt_image_paths: list[str],
    device: torch.device,
    *,
    limit: int,
    stats: NormalizationStats,
    threshold: float = 0.5,
) -> list[SampleComparison]:
    """Run the model on the first ``limit`` dataset rows and compare to labels."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    count = min(limit, len(dataset))
    if count == 0:
        raise ValueError("dataset is empty")

    loader = DataLoader(
        Subset(dataset, list(range(count))),
        batch_size=min(count, 16),
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    comparisons: list[SampleComparison] = []
    offset = 0
    with torch.inference_mode():
        for images, attrs in loader:
            logits = model(images.to(device, non_blocking=True))
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = attrs.numpy()
            for row in range(images.shape[0]):
                comparisons.append(
                    compare_sample(
                        image_relpath=gt_image_paths[offset + row],
                        image=images[row],
                        targets=targets[row],
                        probabilities=probs[row],
                        attribute_names=dataset.attribute_names,
                        stats=stats,
                        threshold=threshold,
                    )
                )
            offset += images.shape[0]
    return comparisons


def plot_comparisons(
    comparisons: list[SampleComparison],
    *,
    title: str,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Draw one readable row per sample: image + TP/FP/FN text panel."""
    count = len(comparisons)
    fig, axes = plt.subplots(
        count,
        2,
        figsize=(11.5, max(3.2 * count, 3.2)),
        gridspec_kw={"width_ratios": [1.0, 1.55], "wspace": 0.08},
    )
    rows = np.atleast_2d(axes)
    for row_axes, sample in zip(rows, comparisons, strict=True):
        ax_img, ax_txt = row_axes
        ax_img.imshow(sample.rgb)
        ax_img.set_title(Path(sample.image_relpath).name, fontsize=11)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_txt.axis("off")
        gt = ", ".join(sample.label_positives) or "—"
        pred = ", ".join(sample.pred_positives) or "—"
        tp = ", ".join(sample.true_positives) or "—"
        fp = ", ".join(sample.false_positives) or "—"
        fn = ", ".join(sample.false_negatives) or "—"
        ax_txt.text(
            0.0,
            1.0,
            (
                f"True positives ({len(sample.true_positives)}):\n  {tp}\n\n"
                f"False positives ({len(sample.false_positives)}):\n  {fp}\n\n"
                f"False negatives ({len(sample.false_negatives)}):\n  {fn}\n\n"
                f"GT ({len(sample.label_positives)}):\n  {gt}\n\n"
                f"Pred ({len(sample.pred_positives)}):\n  {pred}"
            ),
            transform=ax_txt.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            linespacing=1.35,
        )
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_comparison_summary(comparisons: list[SampleComparison]) -> None:
    """Print a compact text summary of label-vs-prediction mismatches."""
    for index, sample in enumerate(comparisons):
        print(
            f"[{index}] {sample.image_relpath} "
            f"TP={len(sample.true_positives)} "
            f"FP={len(sample.false_positives)} "
            f"FN={len(sample.false_negatives)}"
        )
        if sample.false_positives:
            print(f"    FP: {', '.join(sample.false_positives)}")
        if sample.false_negatives:
            print(f"    FN: {', '.join(sample.false_negatives)}")
