"""Loading UPAR annotations (gt.csv / queries.csv / ids.csv) and images."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from src.wacv.attributes import attribute_names_from_columns

KNOWN_DOMAINS: tuple[str, ...] = ("PA100k", "PETA", "Market1501")
DEFAULT_IMAGE_SIZE: tuple[int, int] = (256, 128)

# open_clip openai mean/std; aspect-preserving ReID crop (default 256x128).
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


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
) -> transforms.Compose:
    """Pick ImageNet or CLIP preprocessing at ``image_size`` (H, W)."""
    size = (int(image_size[0]), int(image_size[1]))
    if model_name.startswith("clip_"):
        if train:
            return _train_transform(size, mean=_CLIP_MEAN, std=_CLIP_STD)
        return _eval_transform(size, mean=_CLIP_MEAN, std=_CLIP_STD)
    if train:
        return _train_transform(size, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return _eval_transform(size, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


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
        query_ids: np.ndarray | None = None,
    ) -> None:
        self._gt = gt.reset_index(drop=True)
        self._images_root = Path(images_root)
        self._transform = transform
        self._attribute_names = attribute_names or attribute_names_from_columns(
            tuple(str(c) for c in self._gt.columns)
        )
        if query_ids is not None and len(query_ids) != len(self._gt):
            raise ValueError(
                f"query_ids length {len(query_ids)} != dataset length {len(self._gt)}"
            )
        self._query_ids = (
            None if query_ids is None else np.asarray(query_ids, dtype=np.int64)
        )

    @property
    def attribute_names(self) -> tuple[str, ...]:
        return self._attribute_names

    @property
    def query_ids(self) -> np.ndarray | None:
        """Per-row task-2 query ids, or ``None`` when not attached."""
        return self._query_ids

    def __len__(self) -> int:
        return len(self._gt)

    def __getitem__(
        self, index: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
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
        return (
            transformed,
            attrs,
            torch.tensor(int(self._query_ids[index]), dtype=torch.long),
        )


class QueryPkSampler(Sampler[list[int]]):
    """P×K batch sampler: ``P`` query ids × ``K`` images sharing that id.

    Yields batches of size ``P * K``. Query ids with fewer than ``K`` images
    are skipped. Each epoch reshuffles identity order and within-id picks.
    """

    def __init__(
        self,
        query_ids: Sequence[int],
        *,
        p: int = 16,
        k: int = 4,
        seed: int = 0,
    ) -> None:
        if p < 1 or k < 1:
            raise ValueError(f"p and k must be >= 1, got p={p}, k={k}")
        self.p = int(p)
        self.k = int(k)
        self.seed = int(seed)
        self._epoch = 0
        by_id: dict[int, list[int]] = defaultdict(list)
        for index, query_id in enumerate(query_ids):
            by_id[int(query_id)].append(index)
        self._by_id = {
            query_id: indices
            for query_id, indices in by_id.items()
            if len(indices) >= self.k
        }
        if len(self._by_id) < self.p:
            raise ValueError(
                f"Need at least P={self.p} query ids with >=K={self.k} images, "
                f"got {len(self._by_id)}"
            )

    def __len__(self) -> int:
        return len(self._by_id) // self.p

    def set_epoch(self, epoch: int) -> None:
        """Advance the per-epoch shuffle seed (call once per epoch)."""
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        identity_ids = list(self._by_id.keys())
        rng.shuffle(identity_ids)
        usable = len(identity_ids) - (len(identity_ids) % self.p)
        for start in range(0, usable, self.p):
            batch: list[int] = []
            for query_id in identity_ids[start : start + self.p]:
                members = self._by_id[query_id]
                batch.extend(rng.sample(members, self.k))
            yield batch
