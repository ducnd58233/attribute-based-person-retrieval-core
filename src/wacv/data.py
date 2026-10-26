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
    DEFAULT_RANDOM_ERASING_P,
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


def _clahe_uint8(
    channel: np.ndarray,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> np.ndarray:
    """Clipped histogram equalization on a uint8 2-D channel (tile + bilinear)."""
    height, width = channel.shape
    tiles_y, tiles_x = tile_grid_size
    tile_h = int(np.ceil(height / tiles_y))
    tile_w = int(np.ceil(width / tiles_x))
    pad_h = tile_h * tiles_y - height
    pad_w = tile_w * tiles_x - width
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")
    maps = np.zeros((tiles_y, tiles_x, 256), dtype=np.float32)
    tile_pixels = float(tile_h * tile_w)
    clip_threshold = max(1.0, clip_limit * tile_pixels / 256.0)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile = padded[
                ty * tile_h : (ty + 1) * tile_h,
                tx * tile_w : (tx + 1) * tile_w,
            ]
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float32)
            excess = np.maximum(hist - clip_threshold, 0.0).sum()
            hist = np.minimum(hist, clip_threshold)
            hist += excess / 256.0
            cdf = hist.cumsum()
            positive = cdf[cdf > 0]
            cdf_min = float(positive.min()) if positive.size else 0.0
            denom = max(tile_pixels - cdf_min, 1.0)
            maps[ty, tx] = np.clip(((cdf - cdf_min) / denom) * 255.0, 0.0, 255.0)
    ys = (np.arange(height, dtype=np.float32) + 0.5) * tiles_y / height - 0.5
    xs = (np.arange(width, dtype=np.float32) + 0.5) * tiles_x / width - 0.5
    ys = np.clip(ys, 0.0, tiles_y - 1.001)
    xs = np.clip(xs, 0.0, tiles_x - 1.001)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, tiles_y - 1)
    x1 = np.minimum(x0 + 1, tiles_x - 1)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]
    src = channel.astype(np.int32)
    y0g = y0[:, None]
    y1g = y1[:, None]
    x0g = x0[None, :]
    x1g = x1[None, :]
    top = maps[y0g, x0g, src] * (1.0 - wx) + maps[y0g, x1g, src] * wx
    bot = maps[y1g, x0g, src] * (1.0 - wx) + maps[y1g, x1g, src] * wx
    out = top * (1.0 - wy) + bot * wy
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


class RandomCLAHE:
    """Probabilistic CLAHE on the PIL LAB L-channel (train-only; no OpenCV)."""

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        p: float = 0.5,
    ) -> None:
        self._clip_limit = float(clip_limit)
        self._tile_grid_size = (int(tile_grid_size[0]), int(tile_grid_size[1]))
        self._p = float(p)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self._p:
            return image
        lab = np.asarray(image.convert("LAB"), dtype=np.uint8).copy()
        lab[:, :, 0] = _clahe_uint8(
            lab[:, :, 0], self._clip_limit, self._tile_grid_size
        )
        return Image.fromarray(lab, mode="LAB").convert("RGB")


# T28 / E81: fixed V-deg severities (organizer val only; no hue ops).
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
        reduced = image.resize((small_w, small_h), Image.Resampling.BILINEAR)
        return reduced.resize((width, height), Image.Resampling.BILINEAR)


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
    mean: Sequence[float],
    std: Sequence[float],
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
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


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


def _pre_resize_steps(augment: DataConfig | None) -> list[object]:
    """Optional train-only steps on the full-size image, in a fixed order."""
    if augment is None:
        return []
    steps: list[object] = []
    contrast = augment.autocontrast_equalize
    if contrast.enabled:
        steps += [
            transforms.RandomAutocontrast(p=contrast.autocontrast_p),
            transforms.RandomEqualize(p=contrast.equalize_p),
        ]
    clahe = augment.clahe
    if clahe.enabled:
        steps.append(
            RandomCLAHE(
                clip_limit=clahe.clip_limit,
                tile_grid_size=clahe.tile_grid_size,
                p=clahe.p,
            )
        )
    affine = augment.random_affine
    if affine.enabled:
        steps.append(
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=affine.degrees,
                        translate=(affine.translate, affine.translate),
                        scale=affine.scale,
                    )
                ],
                p=affine.p,
            )
        )
    return steps


def _policy_step(augment: DataConfig | None) -> object:
    """AugMix, unless the config picks RandAugment or TrivialAugmentWide."""
    if augment is not None and augment.rand_augment.enabled:
        policy = augment.rand_augment
        return transforms.RandAugment(
            num_ops=policy.num_ops, magnitude=policy.magnitude
        )
    if augment is not None and augment.trivial_augment.enabled:
        return transforms.TrivialAugmentWide()
    return transforms.AugMix()


def _random_erasing_p(augment: DataConfig | None) -> float:
    if augment is None:
        return DEFAULT_RANDOM_ERASING_P
    return augment.random_erasing_p


def _train_transform(
    size: tuple[int, int],
    stats: NormalizationStats,
    augment: DataConfig | None = None,
) -> transforms.Compose:
    return transforms.Compose(
        [
            *_pre_resize_steps(augment),
            transforms.Resize(size),
            _policy_step(augment),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=stats.mean, std=stats.std),
            transforms.RandomErasing(p=_random_erasing_p(augment)),
        ]
    )


def _train_transform_degrade(
    size: tuple[int, int],
    degrade: DegradeAugConfig,
    stats: NormalizationStats,
    augment: DataConfig | None = None,
) -> transforms.Compose:
    scale_lo, scale_hi = degrade.downscale
    jpeg_lo, jpeg_hi = degrade.jpeg
    return transforms.Compose(
        [
            *_pre_resize_steps(augment),
            transforms.Resize(size),
            RandomDownscaleUpsample(scale_lo, scale_hi),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=degrade.blur_p,
            ),
            RandomJpegCompression(jpeg_lo, jpeg_hi),
            _policy_step(augment),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=stats.mean, std=stats.std),
            transforms.RandomErasing(p=_random_erasing_p(augment)),
        ]
    )


def transforms_for_model(
    model_name: str,
    *,
    train: bool,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    degrade_aug: DegradeAugConfig | None = None,
    val_degrade: str | None = None,
    augment: DataConfig | None = None,
) -> transforms.Compose:
    """Pick the model family's preprocessing at ``image_size`` (H, W).

    ``augment`` supplies the train-only options (CLAHE, RandAugment,
    TrivialAugmentWide, RandomAffine, AutoContrast + Equalize, the Random
    Erasing probability); ``None`` keeps the default pipeline.
    """
    stats = normalization_for_model(model_name)
    if val_degrade is not None and train:
        raise ValueError("val_degrade is eval-only; refuse train=True")
    if not train:
        if val_degrade is not None:
            return val_degrade_transform(
                image_size, val_degrade, mean=stats.mean, std=stats.std
            )
        return _eval_transform(image_size, stats)
    if degrade_aug is not None and degrade_aug.enabled:
        return _train_transform_degrade(image_size, degrade_aug, stats, augment)
    return _train_transform(image_size, stats, augment)


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
        frame = gt.reset_index(drop=True)
        self._images_root = Path(images_root)
        self._transform = transform
        self._attribute_names = attribute_names or attribute_names_from_columns(
            tuple(str(c) for c in frame.columns)
        )
        # Plain arrays instead of a DataFrame: no pandas row lookup per item,
        # and a cheaper pickle into each spawned DataLoader worker.
        self._images = tuple(str(image) for image in frame["image"])
        self._labels = frame[list(self._attribute_names)].to_numpy(dtype=np.float32)
        if query_id_by_image is not None:
            self._query_ids = [
                int(query_id_by_image.get(image, -1)) for image in self._images
            ]
        else:
            self._query_ids = None

    @property
    def attribute_names(self) -> tuple[str, ...]:
        return self._attribute_names

    @property
    def images(self) -> tuple[str, ...]:
        """Image paths relative to the data root, in item order."""
        return self._images

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, int]:
        image = Image.open(self._images_root / self._images[index]).convert("RGB")
        transformed = self._transform(image)
        if not isinstance(transformed, torch.Tensor):
            raise TypeError("AttributeDataset transforms must return a tensor")
        attrs = torch.tensor(self._labels[index])
        if self._query_ids is None:
            return transformed, attrs
        return transformed, attrs, self._query_ids[index]


def build_eval_loader(
    gt: pd.DataFrame,
    attribute_names: tuple[str, ...],
    model_name: str,
    data_config: DataConfig,
    device: torch.device,
    *,
    val_degrade: str | None = None,
) -> DataLoader:
    """Un-shuffled eval loader over ``gt`` with the model family's eval transform."""
    dataset = AttributeDataset(
        gt,
        data_config.root,
        transforms_for_model(
            model_name,
            train=False,
            image_size=data_config.image_size,
            val_degrade=val_degrade,
        ),
        attribute_names=attribute_names,
    )
    return build_loader(dataset, data_config, device, training=False)


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
    training: bool,
) -> DataLoader:
    """DataLoader for training (shuffled) or evaluation (in order).

    Training uses ``batch_size`` and keeps its workers alive across epochs.
    Evaluation uses the larger ``eval_batch_size`` and lets its workers exit
    after each pass, so a val loader does not hold a second set of worker
    processes in memory for the whole run. Torch seeds ``random``, numpy and
    torch in every worker itself.
    """
    workers = resolve_num_workers(data_config.num_workers)
    return DataLoader(
        dataset,
        batch_size=data_config.batch_size if training else data_config.eval_batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=training and workers > 0,
    )
