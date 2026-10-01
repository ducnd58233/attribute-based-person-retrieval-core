"""Loading UPAR annotations (gt.csv / queries.csv / ids.csv) and images."""

from __future__ import annotations

import random
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.wacv.attributes import attribute_names_from_columns
from src.wacv.config import DegradeAugConfig

KNOWN_DOMAINS: tuple[str, ...] = ("PA100k", "PETA", "Market1501")
DEFAULT_IMAGE_SIZE: tuple[int, int] = (256, 128)

# open_clip openai mean/std; aspect-preserving ReID crop (default 256x128).
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomDownscaleUpsample:
    """Randomly shrink then restore spatial size (simulates low-res CCTV)."""

    def __init__(
        self,
        scale_min: float,
        scale_max: float,
    ) -> None:
        self._scale_min = scale_min
        self._scale_max = scale_max

    def __call__(self, image: Image.Image) -> Image.Image:
        scale = random.uniform(self._scale_min, self._scale_max)
        width, height = image.size
        small_w = max(1, int(width * scale))
        small_h = max(1, int(height * scale))
        reduced = image.resize((small_w, small_h), Image.BILINEAR)
        return reduced.resize((width, height), Image.BILINEAR)


class RandomJpegCompression:
    """Re-encode through JPEG at a random quality (simulates compression artifacts)."""

    def __init__(self, quality_min: int, quality_max: int) -> None:
        self._quality_min = quality_min
        self._quality_max = quality_max

    def __call__(self, image: Image.Image) -> Image.Image:
        quality = random.randint(self._quality_min, self._quality_max)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


# V-deg fixed severities (HYPOTHESIS next-step): no colour ops; deterministic.
VAL_DEGRADE_LEVELS: dict[str, dict[str, float | int]] = {
    "L1": {"downscale": 0.60, "blur_sigma": 0.6, "jpeg_quality": 75},
    "L2": {"downscale": 0.40, "blur_sigma": 1.0, "jpeg_quality": 55},
    "L3": {"downscale": 0.28, "blur_sigma": 1.4, "jpeg_quality": 35},
}


class FixedDownscaleUpsample:
    """Deterministic shrink-then-restore (fixed scale)."""

    def __init__(self, scale: float) -> None:
        if not (0.0 < scale <= 1.0):
            raise ValueError(f"scale must be in (0, 1], got {scale}")
        self._scale = float(scale)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        small_w = max(1, int(width * self._scale))
        small_h = max(1, int(height * self._scale))
        reduced = image.resize((small_w, small_h), Image.BILINEAR)
        return reduced.resize((width, height), Image.BILINEAR)


class FixedJpegCompression:
    """Deterministic JPEG re-encode at a fixed quality."""

    def __init__(self, quality: int) -> None:
        if not (1 <= int(quality) <= 100):
            raise ValueError(f"jpeg quality must be 1..100, got {quality}")
        self._quality = int(quality)

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self._quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def val_degrade_transform(
    size: tuple[int, int],
    level: str,
    *,
    mean: list[float],
    std: list[float],
) -> transforms.Compose:
    """Eval transform with fixed V-deg severity (no hue/saturation ops)."""
    if level not in VAL_DEGRADE_LEVELS:
        raise ValueError(f"Unknown val degrade level {level!r}; expected L1/L2/L3")
    spec = VAL_DEGRADE_LEVELS[level]
    scale = float(spec["downscale"])
    blur_sigma = float(spec["blur_sigma"])
    jpeg_q = int(spec["jpeg_quality"])
    return transforms.Compose(
        [
            transforms.Resize(size),
            FixedDownscaleUpsample(scale),
            transforms.GaussianBlur(kernel_size=3, sigma=(blur_sigma, blur_sigma)),
            FixedJpegCompression(jpeg_q),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def color_jitter_hue_span(
    pipeline: transforms.Compose,
) -> tuple[float, float] | None:
    """Return the (min, max) hue jitter factors in ``pipeline``, if any."""
    for step in pipeline.transforms:
        if isinstance(step, transforms.ColorJitter):
            hue = step.hue
            if hue is None:
                return (0.0, 0.0)
            if isinstance(hue, (list, tuple)) and len(hue) == 2:
                return (float(hue[0]), float(hue[1]))
            return (float(hue), float(hue))
    return None


def _eval_transform(
    size: tuple[int, int], *, mean: list[float], std: list[float]
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _train_transform(
    size: tuple[int, int], *, mean: list[float], std: list[float]
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.AugMix(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25),
        ]
    )


def _train_transform_degrade(
    size: tuple[int, int],
    degrade: DegradeAugConfig,
    *,
    mean: list[float],
    std: list[float],
) -> transforms.Compose:
    scale_lo, scale_hi = degrade.downscale
    jpeg_lo, jpeg_hi = degrade.jpeg
    return transforms.Compose(
        [
            transforms.Resize(size),
            RandomDownscaleUpsample(scale_lo, scale_hi),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=degrade.blur_p,
            ),
            RandomJpegCompression(jpeg_lo, jpeg_hi),
            transforms.AugMix(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25),
        ]
    )


IMAGE_TRANSFORM = _eval_transform(
    DEFAULT_IMAGE_SIZE, mean=_IMAGENET_MEAN, std=_IMAGENET_STD
)
TRAIN_TRANSFORM = _train_transform(
    DEFAULT_IMAGE_SIZE, mean=_IMAGENET_MEAN, std=_IMAGENET_STD
)
CLIP_IMAGE_TRANSFORM = _eval_transform(
    DEFAULT_IMAGE_SIZE, mean=_CLIP_MEAN, std=_CLIP_STD
)
CLIP_TRAIN_TRANSFORM = _train_transform(
    DEFAULT_IMAGE_SIZE, mean=_CLIP_MEAN, std=_CLIP_STD
)


def transforms_for_model(
    model_name: str,
    *,
    train: bool,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    degrade_aug: DegradeAugConfig | None = None,
    val_degrade: str | None = None,
) -> transforms.Compose:
    """Pick ImageNet or CLIP preprocessing at ``image_size`` (H, W)."""
    size = (int(image_size[0]), int(image_size[1]))
    use_degrade = train and degrade_aug is not None and degrade_aug.enabled
    if val_degrade is not None and train:
        raise ValueError("val_degrade is eval-only; refuse train=True")
    if model_name.startswith("clip_"):
        mean, std = _CLIP_MEAN, _CLIP_STD
        if train:
            if use_degrade:
                return _train_transform_degrade(size, degrade_aug, mean=mean, std=std)
            return _train_transform(size, mean=mean, std=std)
        if val_degrade is not None:
            return val_degrade_transform(size, val_degrade, mean=mean, std=std)
        return _eval_transform(size, mean=mean, std=std)
    mean, std = _IMAGENET_MEAN, _IMAGENET_STD
    if train:
        if use_degrade:
            return _train_transform_degrade(size, degrade_aug, mean=mean, std=std)
        return _train_transform(size, mean=mean, std=std)
    if val_degrade is not None:
        return val_degrade_transform(size, val_degrade, mean=mean, std=std)
    return _eval_transform(size, mean=mean, std=std)


def load_gt(csv_path: Path) -> pd.DataFrame:
    """Load a gt.csv: one row per image, `image` plus the attribute columns."""
    header = Path(csv_path).read_text(encoding="utf-8").splitlines()[0]
    columns = [c.strip() for c in header.lstrip("# ").split(",")]
    return pd.read_csv(csv_path, names=columns, comment="#")


def load_queries(csv_path: Path) -> np.ndarray:
    """Load a queries.csv into a `(num_queries, num_attributes)` int array."""
    return pd.read_csv(csv_path, comment="#", header=None).to_numpy(dtype=np.int64)


def load_query_ids(csv_path: Path) -> pd.DataFrame:
    """Load an ids.csv: `image`, `query_id` (row index into that split's queries)."""
    return pd.read_csv(csv_path, names=["image", "query_id"], header=None)


def domain_from_image(image_path: str) -> str:
    """Return the dataset domain from an `image` column value (first path segment)."""
    domain = Path(str(image_path).replace("\\", "/")).parts[0]
    if domain not in KNOWN_DOMAINS:
        raise ValueError(
            f"Unknown domain {domain!r} in image path {image_path!r}; "
            f"expected one of {KNOWN_DOMAINS}"
        )
    return domain


def filter_gt_by_domains(
    gt: pd.DataFrame, domains: Sequence[str] | None
) -> pd.DataFrame:
    """Keep rows whose `image` domain is in `domains`. ``None`` keeps all rows."""
    if domains is None:
        return gt
    allowed = set(domains)
    unknown = allowed - set(KNOWN_DOMAINS)
    if unknown:
        raise ValueError(f"Unknown train/eval domains: {sorted(unknown)}")
    mask = gt["image"].map(domain_from_image).isin(allowed)
    return gt.loc[mask].reset_index(drop=True)


def build_lodo_m_split(
    gt: pd.DataFrame,
    attribute_names: Sequence[str],
    *,
    holdout_domain: str = "Market1501",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Leave-one-dataset-out Market proxy: holdout gallery + unique query vectors.

    Gallery is the holdout-domain subset of ``gt``. Queries are the unique
    attribute vectors present in that subset, in first-occurrence order.
    """
    gallery = filter_gt_by_domains(gt, [holdout_domain])
    if gallery.empty:
        raise ValueError(
            f"LODO-M gallery is empty for holdout domain {holdout_domain!r}"
        )
    attrs = gallery[list(attribute_names)].to_numpy(dtype=np.int64)
    _, first_index = np.unique(attrs, axis=0, return_index=True)
    queries = attrs[np.sort(first_index)]
    return gallery, queries


class AttributeDataset(Dataset):
    """One pedestrian image and its attribute vector (order from gt columns)."""

    def __init__(
        self,
        gt: pd.DataFrame,
        images_root: Path,
        transform: transforms.Compose = IMAGE_TRANSFORM,
        *,
        attribute_names: tuple[str, ...] | None = None,
    ) -> None:
        self._gt = gt.reset_index(drop=True)
        self._images_root = Path(images_root)
        self._transform = transform
        self._attribute_names = attribute_names or attribute_names_from_columns(
            tuple(str(c) for c in self._gt.columns)
        )

    @property
    def attribute_names(self) -> tuple[str, ...]:
        return self._attribute_names

    def __len__(self) -> int:
        return len(self._gt)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._gt.iloc[index]
        image = Image.open(self._images_root / row["image"]).convert("RGB")
        transformed = self._transform(image)
        if not isinstance(transformed, torch.Tensor):
            raise TypeError("AttributeDataset transforms must return a tensor")
        attrs = torch.from_numpy(
            row[list(self._attribute_names)].to_numpy(dtype=np.float32)
        )
        return transformed, attrs
