"""Loading UPAR annotations (gt.csv / queries.csv / ids.csv) and images."""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.wacv.attributes import attribute_names_from_columns
from src.wacv.config import (
    DEFAULT_IMAGE_SIZE,
    KNOWN_DOMAINS,
    DataConfig,
    DegradeAugConfig,
)
from src.wacv.normalization import NormalizationStats, normalization_for_model


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
        reduced = image.resize((small_w, small_h), Image.Resampling.BILINEAR)
        return reduced.resize((width, height), Image.Resampling.BILINEAR)


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


def _eval_transform(
    size: tuple[int, int], stats: NormalizationStats
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=stats.mean, std=stats.std),
        ]
    )


def _train_transform(
    size: tuple[int, int], stats: NormalizationStats
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.AugMix(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=stats.mean, std=stats.std),
            transforms.RandomErasing(p=0.25),
        ]
    )


def _train_transform_degrade(
    size: tuple[int, int],
    degrade: DegradeAugConfig,
    stats: NormalizationStats,
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
            transforms.Normalize(mean=stats.mean, std=stats.std),
            transforms.RandomErasing(p=0.25),
        ]
    )


def transforms_for_model(
    model_name: str,
    *,
    train: bool,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    degrade_aug: DegradeAugConfig | None = None,
) -> transforms.Compose:
    """Pick the model family's preprocessing at ``image_size`` (H, W)."""
    stats = normalization_for_model(model_name)
    if not train:
        return _eval_transform(image_size, stats)
    if degrade_aug is not None and degrade_aug.enabled:
        return _train_transform_degrade(image_size, degrade_aug, stats)
    return _train_transform(image_size, stats)


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


def load_task2_train_query_supervision(
    data_root: Path,
) -> tuple[dict[str, int], np.ndarray]:
    """Task-2 train split only: image -> query_id and full query attribute matrix."""
    train_root = data_root / "annotations" / "task2" / "train"
    ids_path = train_root / "ids.csv"
    queries_path = train_root / "queries.csv"
    if not ids_path.is_file() or not queries_path.is_file():
        raise FileNotFoundError(
            f"Task-2 train ids/queries missing under {train_root}; "
            "ASMR aux requires task2/train/ids.csv and queries.csv"
        )
    ids_df = load_query_ids(ids_path)
    queries = load_queries(queries_path)
    mapping = {
        str(image): int(query_id)
        for image, query_id in zip(ids_df["image"], ids_df["query_id"], strict=True)
    }
    return mapping, queries


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
    mask = gt["image"].map(domain_from_image).isin(sorted(allowed))
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
        transform: transforms.Compose,
        *,
        attribute_names: tuple[str, ...] | None = None,
        query_id_by_image: dict[str, int] | None = None,
    ) -> None:
        self._gt = gt.reset_index(drop=True)
        self._images_root = Path(images_root)
        self._transform = transform
        self._attribute_names = attribute_names or attribute_names_from_columns(
            tuple(str(c) for c in self._gt.columns)
        )
        if query_id_by_image is not None:
            self._query_ids = [
                int(query_id_by_image.get(str(row["image"]), -1))
                for _, row in self._gt.iterrows()
            ]
        else:
            self._query_ids = None

    @property
    def attribute_names(self) -> tuple[str, ...]:
        return self._attribute_names

    def __len__(self) -> int:
        return len(self._gt)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, int]:
        row = self._gt.iloc[index]
        image = Image.open(self._images_root / row["image"]).convert("RGB")
        transformed = self._transform(image)
        if not isinstance(transformed, torch.Tensor):
            raise TypeError("AttributeDataset transforms must return a tensor")
        attrs = torch.from_numpy(
            row[list(self._attribute_names)].to_numpy(dtype=np.float32)
        )
        if self._query_ids is None:
            return transformed, attrs
        return transformed, attrs, self._query_ids[index]


def resolve_num_workers(requested: int) -> int:
    """Cap DataLoader workers to ``cpu_count - 1`` (or 0 when requested <= 0)."""
    if requested <= 0:
        return 0
    cpu_count = os.cpu_count() or 1
    return min(requested, max(cpu_count - 1, 0))


def build_loader(
    dataset: Dataset,
    data_config: DataConfig,
    device: torch.device,
    *,
    shuffle: bool,
) -> DataLoader:
    """DataLoader with the batch size and worker policy of ``data_config``."""
    workers = resolve_num_workers(data_config.num_workers)
    return DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
