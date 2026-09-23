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

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


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


def tensor_to_display_rgb(image: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalize and return an HxWx3 float array in [0, 1]."""
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
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
    threshold: float = 0.5,
) -> SampleComparison:
    """Build a label-vs-prediction comparison for a single sample."""
    predicted = probabilities >= threshold
    label_set = set(positive_names(targets.astype(bool), attribute_names))
    pred_set = set(positive_names(predicted, attribute_names))
    return SampleComparison(
        image_relpath=image_relpath,
        rgb=tensor_to_display_rgb(image),
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
    """Draw a grid of images with GT / pred / error attribute lists."""
    count = len(comparisons)
    columns = min(5, count)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 4.2 * rows))
    flat = np.atleast_1d(axes).ravel()
    for axis, sample in zip(flat, comparisons, strict=False):
        axis.imshow(sample.rgb)
        axis.set_title(Path(sample.image_relpath).name, fontsize=8)
        caption = (
            f"GT ({len(sample.label_positives)}): "
            f"{', '.join(sample.label_positives) or '—'}\n"
            f"Pred ({len(sample.pred_positives)}): "
            f"{', '.join(sample.pred_positives) or '—'}\n"
            f"FP: {', '.join(sample.false_positives) or '—'}\n"
            f"FN: {', '.join(sample.false_negatives) or '—'}"
        )
        axis.set_xlabel(caption, fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in flat[count:]:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
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
