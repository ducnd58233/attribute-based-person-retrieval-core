"""Loading UPAR annotations (gt.csv / queries.csv / ids.csv) and images."""

from __future__ import annotations

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


class DomainBalancedBatchSampler(Sampler[list[int]]):
    """Yield batches with an equal number of samples from each domain.

    Requires ``batch_size`` divisible by the number of distinct domains and
    each domain to hold at least ``batch_size // n_domains`` samples. Shorter
    domains wrap (re-shuffled) so the longest domain is covered once per epoch.
    """

    def __init__(
        self,
        domains: Sequence[str],
        batch_size: int,
        *,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        by_domain: dict[str, list[int]] = defaultdict(list)
        for index, domain in enumerate(domains):
            by_domain[str(domain)].append(index)
        if len(by_domain) < 2:
            raise ValueError(
                "DomainBalancedBatchSampler needs at least two domains, "
                f"got {sorted(by_domain)}"
            )
        if batch_size % len(by_domain) != 0:
            raise ValueError(
                f"batch_size {batch_size} must be divisible by "
                f"{len(by_domain)} domains"
            )
        self._domain_order = tuple(sorted(by_domain))
        self._indices = {
            domain: list(by_domain[domain]) for domain in self._domain_order
        }
        self._per_domain = batch_size // len(self._domain_order)
        self._seed = seed
        self._epoch = 0
        too_small = {
            domain: len(indices)
            for domain, indices in self._indices.items()
            if len(indices) < self._per_domain
        }
        if too_small:
            raise ValueError(
                "Each domain needs at least "
                f"{self._per_domain} samples; too small: {too_small}"
            )
        longest = max(len(indices) for indices in self._indices.values())
        self._num_batches = longest // self._per_domain

    @property
    def domain_order(self) -> tuple[str, ...]:
        """Sorted domain names used when assembling each batch."""
        return self._domain_order

    @property
    def per_domain(self) -> int:
        """Number of samples taken from each domain in a batch."""
        return self._per_domain

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        shuffled = {
            domain: rng.permutation(indices).tolist()
            for domain, indices in self._indices.items()
        }
        cursor = dict.fromkeys(self._domain_order, 0)
        for _ in range(self._num_batches):
            batch: list[int] = []
            for domain in self._domain_order:
                pool = shuffled[domain]
                for _ in range(self._per_domain):
                    if cursor[domain] >= len(pool):
                        shuffled[domain] = rng.permutation(
                            self._indices[domain]
                        ).tolist()
                        cursor[domain] = 0
                        pool = shuffled[domain]
                    batch.append(pool[cursor[domain]])
                    cursor[domain] += 1
            yield batch


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
