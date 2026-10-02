"""Input normalisation statistics per model family (single source of truth)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizationStats:
    """Per-channel RGB mean and standard deviation."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]


IMAGENET = NormalizationStats(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
# open_clip's OpenAI CLIP preprocessing.
CLIP = NormalizationStats(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
)


def normalization_for_model(model_name: str) -> NormalizationStats:
    """CLIP towers use CLIP statistics; every other backbone uses ImageNet's."""
    return CLIP if model_name.startswith("clip_") else IMAGENET
