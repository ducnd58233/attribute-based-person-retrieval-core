"""Codabench Track-2 entry: gallery attribute probs, then configured distances.

Self-contained for the eval image (torch / torchvision / numpy / Pillow).
Weights live under assets/ (written by scripts/package_submission.py).
"""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import convnext_base, convnext_tiny, resnet50

HERE = Path(__file__).resolve().parent
WEIGHTS_PATH = HERE / "assets" / "model.pt"
CARD_PATH = HERE / "assets" / "model_card.json"
CALIBRATION_PATH = HERE / "assets" / "calibration.json"
BATCH_SIZE = 64
_PROB_EPS = 1e-4

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_STATE: dict[str, Any] = {"model": None, "device": None, "card": None}


def _image_transform_from_card(card: dict[str, Any]) -> transforms.Compose:
    size = card.get("image_size", [256, 128])
    if not isinstance(size, list) or len(size) != 2:
        size = [256, 128]
    mean = card.get("image_mean", [0.485, 0.456, 0.406])
    std = card.get("image_std", [0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize((int(size[0]), int(size[1]))),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


class _HasFc(Protocol):
    fc: nn.Module


class _ConvNeXtLike(Protocol):
    features: nn.Module
    avgpool: nn.Module
    classifier: nn.Sequential


class _ResNetLike(Protocol):
    conv1: nn.Module
    bn1: nn.Module
    relu: nn.Module
    maxpool: nn.Module
    layer1: nn.Module
    layer2: nn.Module
    layer3: nn.Module
    layer4: nn.Module
    avgpool: nn.Module


def _classification_head(
    in_features: int, num_attributes: int, dropout: float
) -> nn.Module:
    if dropout == 0.0:
        return nn.Linear(in_features, num_attributes)
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_attributes))


class _HardQueryAdapter(nn.Module):
    def __init__(
        self, *, num_attributes: int, feature_dim: int, adapter_dim: int
    ) -> None:
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(num_attributes, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim),
        )
        self.gallery_proj = nn.Sequential(
            nn.Linear(feature_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim),
        )

    def encode_queries(self, queries: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_proj(queries.float()), dim=-1)

    def encode_gallery_features(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.gallery_proj(features), dim=-1)


class _AttributeModelWithHardQuery(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_attributes: int,
        feature_dim: int,
        adapter_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hard_query_adapter = _HardQueryAdapter(
            num_attributes=num_attributes,
            feature_dim=feature_dim,
            adapter_dim=adapter_dim,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def gallery_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "features") and hasattr(self.backbone, "avgpool"):
            convnext = cast(_ConvNeXtLike, self.backbone)
            features = convnext.features(images)
            pooled = convnext.avgpool(features)
            pooled = torch.flatten(pooled, 1)
        else:
            resnet = cast(_ResNetLike, self.backbone)
            features = resnet.conv1(images)
            features = resnet.bn1(features)
            features = resnet.relu(features)
            features = resnet.maxpool(features)
            features = resnet.layer1(features)
            features = resnet.layer2(features)
            features = resnet.layer3(features)
            features = resnet.layer4(features)
            pooled = torch.flatten(resnet.avgpool(features), 1)
        return self.hard_query_adapter.encode_gallery_features(pooled)

    def query_embeddings(self, queries: torch.Tensor) -> torch.Tensor:
        return self.hard_query_adapter.encode_queries(queries)


def _feature_dim(backbone: nn.Module) -> int:
    if hasattr(backbone, "classifier"):
        head = cast(_ConvNeXtLike, backbone).classifier[-1]
        if isinstance(head, nn.Sequential):
            for module in reversed(list(head.children())):
                if isinstance(module, nn.Linear):
                    return int(module.in_features)
        if isinstance(head, nn.Linear):
            return int(head.in_features)
    if hasattr(backbone, "fc"):
        head = cast(_HasFc, backbone).fc
        if isinstance(head, nn.Sequential):
            for module in reversed(list(head.children())):
                if isinstance(module, nn.Linear):
                    return int(module.in_features)
        if isinstance(head, nn.Linear):
            return int(head.in_features)
    raise TypeError("Could not infer backbone feature dim")


def _build_backbone(card: dict[str, Any]) -> nn.Module:
    name = str(card["name"])
    num_attributes = int(card["num_attributes"])
    dropout = float(card.get("dropout", 0.0))
    if name == "resnet50":
        model = resnet50(weights=None)
        backbone = cast(_HasFc, model)
        if not isinstance(backbone.fc, nn.Linear):
            raise TypeError("Unexpected ResNet50 fc layout")
        backbone.fc = _classification_head(
            backbone.fc.in_features, num_attributes, dropout
        )
        return model
    if name == "convnext_tiny":
        model = convnext_tiny(weights=None)
    elif name == "convnext_base":
        model = convnext_base(weights=None)
    else:
        raise ValueError(f"Unsupported model in model_card.json: {name}")
    output_layer = model.classifier[-1]
    if not isinstance(output_layer, nn.Linear):
        raise TypeError("Unexpected ConvNeXt classifier layout")
    model.classifier[-1] = _classification_head(
        output_layer.in_features, num_attributes, dropout
    )
    return model


def _build_model(card: dict[str, Any]) -> nn.Module:
    backbone = _build_backbone(card)
    if not bool(card.get("hard_query_adapter", False)):
        return backbone
    return _AttributeModelWithHardQuery(
        backbone,
        num_attributes=int(card["num_attributes"]),
        feature_dim=_feature_dim(backbone),
        adapter_dim=int(card.get("adapter_dim", 256)),
    )


def _image_hw_from_card(card: dict[str, Any]) -> tuple[int, int]:
    size = card.get("image_size", [256, 128])
    if not isinstance(size, list) or len(size) != 2:
        return 256, 128
    return int(size[0]), int(size[1])


def _cuda_forward_works(model: nn.Module, card: dict[str, Any]) -> bool:
    """True if a real CUDA forward works (capability mismatches fail here)."""
    if not torch.cuda.is_available():
        return False
    height, width = _image_hw_from_card(card)
    try:
        probe = torch.zeros(1, 3, height, width, device="cuda")
        with torch.inference_mode():
            model(probe)
        torch.cuda.synchronize()
    except RuntimeError:
        return False
    return True


def load_model() -> None:
    """Load weights once. Safe to call repeatedly."""
    if _STATE["model"] is not None:
        return
    with CARD_PATH.open(encoding="utf-8") as handle:
        card = json.load(handle)
    prefer_cuda = torch.cuda.is_available()
    load_device = torch.device("cuda" if prefer_cuda else "cpu")
    if str(card.get("weight_format", "state_dict")) == "torchscript":
        model = torch.jit.load(WEIGHTS_PATH, map_location=load_device)
        if str(card.get("weight_dtype", "fp32")) == "fp16":
            # Stored half to halve the zip; compute stays fp32 (or card autocast).
            model.float()
    else:
        model = _build_model(card)
        state = torch.load(WEIGHTS_PATH, map_location=load_device, weights_only=True)
        model.load_state_dict(state)
        model.to(load_device)
    model.eval()
    device = load_device
    if prefer_cuda and not _cuda_forward_works(model, card):
        model = model.to(torch.device("cpu"))
        device = torch.device("cpu")
    _STATE["model"] = model
    _STATE["device"] = device
    _STATE["card"] = card
    _STATE["transform"] = _image_transform_from_card(card)


def _load_image_tensor(
    item: dict[str, Any], transform: transforms.Compose
) -> torch.Tensor:
    image = Image.open(item["image_path"]).convert("RGB")
    tensor = transform(image)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Image transform must return a tensor")
    return tensor


def _iter_gallery_tensors(
    gallery: list[dict[str, Any]], device: torch.device, *, decode_workers: int = 0
) -> Any:
    """Yield image batches in gallery order.

    With ``decode_workers`` > 0, a thread pool decodes the next batch while the
    caller runs the model on the current one; at most two batches are in flight.
    """
    total = len(gallery)
    transform = _STATE.get("transform", IMAGE_TRANSFORM)
    if not isinstance(transform, transforms.Compose):
        transform = IMAGE_TRANSFORM
    batch_starts = range(0, total, BATCH_SIZE)
    pool = ThreadPoolExecutor(max_workers=decode_workers) if decode_workers else None

    def submit_batch(start: int) -> list[Future[torch.Tensor]]:
        assert pool is not None
        batch = gallery[start : start + BATCH_SIZE]
        return [pool.submit(_load_image_tensor, item, transform) for item in batch]

    try:
        next_batch = submit_batch(0) if pool is not None and total else []
        for start in batch_starts:
            if pool is None:
                batch = gallery[start : start + BATCH_SIZE]
                tensors = [_load_image_tensor(item, transform) for item in batch]
            else:
                current_batch = next_batch
                if start + BATCH_SIZE < total:
                    next_batch = submit_batch(start + BATCH_SIZE)
                tensors = [future.result() for future in current_batch]
            images = torch.stack(tensors, dim=0).to(device, non_blocking=True)
            done = min(start + BATCH_SIZE, total)
            print(f"[rank_gallery] {done}/{total} gallery images processed", flush=True)
            yield images
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


def _autocast(device: torch.device, card: dict[str, Any]) -> Any:
    """Half-precision autocast on CUDA when the card asks for it."""
    amp = str(card.get("inference_amp", "none"))
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if amp not in dtypes:
        raise ValueError(f"Unknown inference_amp {amp!r}; expected none/bf16/fp16")
    return torch.autocast(device_type="cuda", dtype=dtypes[amp])


def _apply_tta_view(
    images: torch.Tensor, view: str, *, scale: float = 1.15
) -> torch.Tensor:
    if view == "identity":
        return images
    if view == "hflip":
        return torch.flip(images, dims=(-1,))
    if view == "scale_crop":
        _, _, height, width = images.shape
        factor = float(scale)
        new_h = max(height, round(height * factor))
        new_w = max(width, round(width * factor))
        upsampled = F.interpolate(
            images, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        top = max(0, (new_h - height) // 2)
        left = max(0, (new_w - width) // 2)
        return upsampled[:, :, top : top + height, left : left + width]
    raise ValueError(f"Unknown TTA view {view!r}")


def _predict_gallery(
    gallery: list[dict[str, Any]], *, with_features: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Probabilities, and identity-view pooled features when ``with_features``."""
    load_model()
    model = _STATE["model"]
    device = _STATE["device"]
    card = _STATE["card"] if isinstance(_STATE["card"], dict) else {}
    if not isinstance(model, nn.Module) or not isinstance(device, torch.device):
        raise RuntimeError("Model failed to load")
    raw_tta = card.get("inference_tta", ["identity"])
    tta_views: list[str]
    if isinstance(raw_tta, list) and raw_tta:
        tta_views = [str(v) for v in raw_tta]
    else:
        tta_views = ["identity"]
    if with_features and "identity" not in tta_views:
        raise ValueError("neighbour_smooth needs the identity view in inference_tta")
    tta_scale = float(card.get("inference_tta_scale", 1.15))
    batches: list[np.ndarray] = []
    feature_batches: list[np.ndarray] = []
    with torch.inference_mode(), _autocast(device, card):
        for images in _iter_gallery_tensors(
            gallery, device, decode_workers=int(card.get("inference_decode_workers", 0))
        ):
            probs_sum: torch.Tensor | None = None
            for view in tta_views:
                output = model(_apply_tta_view(images, view, scale=tta_scale))
                logits = output[0] if isinstance(output, tuple) else output
                if with_features and view == "identity":
                    if not isinstance(output, tuple):
                        raise TypeError("model must return (logits, features)")
                    feature_batches.append(output[1].float().cpu().numpy())
                probs = torch.sigmoid(logits.float())
                probs_sum = probs if probs_sum is None else probs_sum + probs
            assert probs_sum is not None
            batches.append((probs_sum / float(len(tta_views))).float().cpu().numpy())
    if not batches:
        raise ValueError("Gallery is empty")
    features = np.concatenate(feature_batches, axis=0) if with_features else None
    return np.concatenate(batches, axis=0), features


def _neighbour_smooth(
    probabilities: np.ndarray, features: np.ndarray, smooth: dict[str, Any]
) -> np.ndarray:
    """Blend each image with its nearest gallery neighbours (cosine, gated).

    Mirrors ``src/wacv/evaluation/smoothing.py``; this file ships alone.
    """
    count = len(probabilities)
    k = min(int(smooth["k"]), count - 1)
    if k < 1:
        return probabilities
    beta = float(smooth["beta"])
    gate = float(smooth["min_similarity"])
    unit = features.astype(np.float32)
    unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    smoothed = probabilities.copy()
    chunk = 2048
    for start in range(0, count, chunk):
        similarity = unit[start : start + chunk] @ unit.T
        rows = np.arange(similarity.shape[0])
        similarity[rows, start + rows] = -np.inf
        nearest = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        kept = np.take_along_axis(similarity, nearest, axis=1) >= gate
        weights = kept.astype(np.float32)
        kept_count = weights.sum(axis=1)
        has_neighbours = kept_count > 0
        neighbour_sum = (probabilities[nearest] * weights[..., None]).sum(axis=1)
        neighbour_mean = (
            neighbour_sum[has_neighbours] / kept_count[has_neighbours, None]
        )
        block = smoothed[start : start + similarity.shape[0]]
        block[has_neighbours] = (1.0 - beta) * block[
            has_neighbours
        ] + beta * neighbour_mean
    return smoothed


def _predict_gallery_embeddings(gallery: list[dict[str, Any]]) -> np.ndarray:
    load_model()
    model = _STATE["model"]
    device = _STATE["device"]
    if not isinstance(model, _AttributeModelWithHardQuery):
        raise TypeError("clear_hard requires hard_query_adapter weights")
    if not isinstance(device, torch.device):
        raise RuntimeError("Model failed to load")
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images in _iter_gallery_tensors(gallery, device):
            batches.append(model.gallery_embeddings(images).float().cpu().numpy())
    if not batches:
        raise ValueError("Gallery is empty")
    return np.concatenate(batches, axis=0)


def _l1_distance(queries: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    # In place from the matmul on: the naive `a + queries @ b.T` form holds
    # both the matmul result and the sum's output as separate full Q x G
    # arrays at once (called from _hybrid_distance, which is itself holding
    # a same-size accumulator - see its docstring).
    out = queries @ (1.0 - 2.0 * probabilities).T
    out += probabilities.sum(1)[None, :]
    return out


def _specker_distance(
    queries: np.ndarray, probabilities: np.ndarray, temperature: float
) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float32), _PROB_EPS, 1.0 - _PROB_EPS)
    if temperature != 1.0:
        logits = np.log(clipped) - np.log1p(-clipped)
        clipped = 1.0 / (1.0 + np.exp(-(logits / float(temperature))))
    return clipped.sum(1)[None, :] + queries @ (1.0 - 2.0 * clipped).T


def _apply_per_attribute_temperatures(
    probabilities: np.ndarray, temperatures: np.ndarray
) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float32), _PROB_EPS, 1.0 - _PROB_EPS)
    logits = np.log(clipped) - np.log1p(-clipped)
    scaled = logits / np.maximum(temperatures.astype(np.float32)[None, :], 1e-3)
    return 1.0 / (1.0 + np.exp(-scaled))


def _load_calibration_temperatures() -> np.ndarray | None:
    if not CALIBRATION_PATH.is_file():
        return None
    payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    temps = payload.get("temperatures")
    if not isinstance(temps, list) or not temps:
        return None
    return np.asarray(temps, dtype=np.float32)


def _nll_distance(
    queries: np.ndarray, probabilities: np.ndarray, temperature: float
) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float32), _PROB_EPS, 1.0 - _PROB_EPS)
    if temperature != 1.0:
        logits = np.log(clipped) - np.log1p(-clipped)
        clipped = 1.0 / (1.0 + np.exp(-(logits / float(temperature))))
    log_p = np.log(clipped)
    log_1mp = np.log(1.0 - clipped)
    logits = log_p - log_1mp
    # In place from the matmul on, same reason as _l1_distance above.
    out = queries @ logits.T
    out *= -1.0
    out -= log_1mp.sum(axis=1)[None, :]
    return out


_EXCLUSIVE_GROUPS: tuple[tuple[tuple[int, ...], bool], ...] = (
    ((0, 1, 2), False),
    ((4, 5, 6), True),
    (tuple(range(8, 20)), False),
    (tuple(range(21, 33)), False),
    ((33, 34), False),
)


def _bernoulli_nll_bits(
    queries: np.ndarray, calibrated: np.ndarray, indices: tuple[int, ...]
) -> np.ndarray:
    if not indices:
        return np.zeros((queries.shape[0], calibrated.shape[0]), dtype=np.float32)
    idx = list(indices)
    query_bits = queries[:, idx].astype(np.float32)
    probs = calibrated[:, idx]
    log_p = np.log(probs)
    log_1mp = np.log(1.0 - probs)
    logits = log_p - log_1mp
    return (-(query_bits @ logits.T) - log_1mp.sum(axis=1)[None, :]).astype(np.float32)


def _group_categorical_nll_distance(
    queries: np.ndarray, probabilities: np.ndarray, temperature: float
) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float32), _PROB_EPS, 1.0 - _PROB_EPS)
    if temperature != 1.0:
        logits = np.log(clipped) - np.log1p(-clipped)
        clipped = 1.0 / (1.0 + np.exp(-(logits / float(temperature))))
    distance = np.zeros((queries.shape[0], clipped.shape[0]), dtype=np.float32)
    covered: set[int] = set()
    for indices, has_none in _EXCLUSIVE_GROUPS:
        covered.update(indices)
        idx = list(indices)
        query_bits = queries[:, idx].astype(np.float32)
        group_probs = clipped[:, idx]
        if has_none:
            none_mass = np.clip(
                1.0 - group_probs.sum(axis=1, keepdims=True), _PROB_EPS, 1.0
            )
            extended_probs = np.concatenate([group_probs, none_mass], axis=1)
            none_query = (query_bits.sum(axis=1, keepdims=True) == 0).astype(np.float32)
            extended_queries = np.concatenate([query_bits, none_query], axis=1)
        else:
            extended_probs = group_probs
            extended_queries = query_bits
        row_sum = np.maximum(extended_probs.sum(axis=1, keepdims=True), _PROB_EPS)
        renorm = np.clip(extended_probs / row_sum, _PROB_EPS, 1.0)
        categorical = -(extended_queries @ np.log(renorm).T)
        bernoulli = _bernoulli_nll_bits(queries, clipped, indices)
        ones = extended_queries.sum(axis=1) == 1.0
        multi = extended_queries.sum(axis=1) > 1.0
        zero = extended_queries.sum(axis=1) == 0.0
        distance += (
            np.where(ones[:, None], categorical, 0.0)
            + np.where((multi | (zero & (not has_none)))[:, None], bernoulli, 0.0)
        ).astype(np.float32)
    independent = tuple(i for i in range(clipped.shape[1]) if i not in covered)
    distance += _bernoulli_nll_bits(queries, clipped, independent)
    return distance


def _minmax_per_query(distances: np.ndarray) -> np.ndarray:
    """Scale each query row to [0, 1]; constant rows become zeros.

    Mutates and returns ``distances`` itself rather than allocating a fresh
    Q x G array. The original
    `np.where(spread > 0, normalised, np.zeros_like(distances))` held four
    full Q x G float32 arrays at once (the input, ``normalised``, the
    ``zeros_like`` array, and the ``where`` output) - on the real E17
    gallery shape (~3462x33407) that is four ~463MB allocations alive
    simultaneously. Both call sites in `_hybrid_distance` pass a scratch
    array built solely to be normalised here (`_nll_distance`'s and
    `_l1_distance`'s return values), so mutating it is safe; a caller that
    needs its array preserved should pass ``distances.copy()``.
    """
    row_min = distances.min(axis=1, keepdims=True)
    row_max = distances.max(axis=1, keepdims=True)
    spread = row_max - row_min
    safe = np.maximum(spread, 1e-8)
    distances -= row_min
    distances /= safe
    zero_rows = (spread <= 0).ravel()
    if zero_rows.any():
        distances[zero_rows] = 0.0
    return distances


def _hybrid_distance(
    queries: np.ndarray,
    probabilities: np.ndarray,
    temperature: float,
    alpha: float,
) -> np.ndarray:
    """Mix min-max-normalised NLL and L1, one term at a time.

    Computing `nll` and `l1` as separate named arrays and combining them at
    the end (`weight * nll + (1.0 - weight) * l1`) keeps both full Q x G
    arrays alive for the combine step, plus two more for the
    weight-scaled temporaries the `*`/`+` create. Accumulating into `out`
    in place means at most one of {nll, l1} occupies its own Q x G buffer
    at a time.
    """
    weight = float(alpha)
    out = _minmax_per_query(_nll_distance(queries, probabilities, temperature))
    out *= weight
    l1 = _minmax_per_query(_l1_distance(queries, probabilities))
    l1 *= 1.0 - weight
    out += l1
    return out


def _clear_hard_distance(
    queries: np.ndarray,
    probabilities: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    temperature: float,
    clear_fusion: float,
) -> np.ndarray:
    specker = _specker_distance(queries, probabilities, temperature)
    cosine_distance = 1.0 - (
        query_embeddings.astype(np.float32) @ gallery_embeddings.astype(np.float32).T
    )
    fusion = float(clear_fusion)
    return (1.0 - fusion) * specker + fusion * cosine_distance


def _dual_softmax_rerank(
    distances: np.ndarray,
    tau: float,
    *,
    chunk_cols: int = 4096,
) -> np.ndarray:
    """S' = S + log_softmax_queries(S/tau); return -S'.

    Softmax is column-wise (over queries). Process gallery columns in chunks so
    a large QxG matrix does not allocate multi-GB float64 temporaries and get
    the Codabench worker OOM-killed ("Child task failed or non-zero return").

    Mutates and returns its own float32-cast copy of ``distances`` in place
    rather than allocating a second full Q x G output array: on the real E17
    gallery shape (~3462x33407) that second array was itself a ~463MB
    allocation the caller never needed once the chunked loop below can write
    every result straight back over the value it read. Safe because
    ``rank_gallery``'s only call site immediately reassigns its `distances`
    variable to this function's return value and never reads the pre-call
    array again - a caller that needs to keep its original array should pass
    ``distances.copy()``.
    """
    similarities = np.asarray(distances, dtype=np.float32)
    np.negative(similarities, out=similarities)
    scale = max(float(tau), 1e-8)
    n_gallery = similarities.shape[1]
    step = max(int(chunk_cols), 1)
    for start in range(0, n_gallery, step):
        end = min(start + step, n_gallery)
        block = similarities[:, start:end] / scale
        col_max = block.max(axis=0, keepdims=True)
        exp_shifted = np.exp(block - col_max)
        log_partition = np.log(
            np.maximum(exp_shifted.sum(axis=0, keepdims=True), 1e-12)
        )
        log_softmax = block - col_max - log_partition
        similarities[:, start:end] = -(similarities[:, start:end] + log_softmax)
    return similarities


def _score_gallery_distances(
    queries: np.ndarray,
    probabilities: np.ndarray,
    card: dict[str, Any],
    *,
    temperature: float,
    gallery: list[Any],
) -> np.ndarray:
    """Dispatch the configured distance; clear_hard may touch ``gallery``."""
    distance_name = str(card.get("ranking_distance", "l1"))
    if distance_name == "clear_hard":
        load_model()
        model = _STATE["model"]
        device = _STATE["device"]
        if not isinstance(model, _AttributeModelWithHardQuery):
            raise TypeError("clear_hard requires hard_query_adapter")
        if not isinstance(device, torch.device):
            raise RuntimeError("Model failed to load")
        gallery_embeddings = _predict_gallery_embeddings(gallery)
        with torch.inference_mode():
            query_embeddings = (
                model.query_embeddings(
                    torch.as_tensor(queries, dtype=torch.float32, device=device)
                )
                .float()
                .cpu()
                .numpy()
            )
        return _clear_hard_distance(
            queries,
            probabilities,
            gallery_embeddings,
            query_embeddings,
            temperature=temperature,
            clear_fusion=float(card.get("clear_fusion", 0.5)),
        )
    if distance_name == "specker":
        return _specker_distance(queries, probabilities, temperature)
    if distance_name in {"nll", "group_categorical"}:
        use_groups = distance_name == "group_categorical" or (
            str(card.get("ranking_group_mode", "independent")) == "categorical"
        )
        if use_groups:
            return _group_categorical_nll_distance(queries, probabilities, temperature)
        return _nll_distance(queries, probabilities, temperature)
    if distance_name == "hybrid":
        return _hybrid_distance(
            queries,
            probabilities,
            temperature,
            float(card.get("ranking_alpha", 0.5)),
        )
    return _l1_distance(queries, probabilities)


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    """Return distances between query vectors and gallery probabilities."""
    load_model()
    card = _STATE["card"] if isinstance(_STATE["card"], dict) else {}
    smooth = card.get("neighbour_smooth")
    probabilities, features = _predict_gallery(
        sample["gallery"], with_features=isinstance(smooth, dict)
    )
    if isinstance(smooth, dict) and features is not None:
        probabilities = _neighbour_smooth(probabilities, features, smooth)
    queries = np.asarray(sample["queries"], dtype=np.float32)
    temperature = float(card.get("ranking_temperature", 1.5))
    if str(card.get("ranking_calibration", "none")) == "per_attribute":
        temps = _load_calibration_temperatures()
        if temps is not None and temps.shape[0] == probabilities.shape[1]:
            probabilities = _apply_per_attribute_temperatures(probabilities, temps)
            temperature = 1.0
    distances = _score_gallery_distances(
        queries,
        probabilities,
        card,
        temperature=temperature,
        gallery=sample["gallery"],
    )
    if str(card.get("ranking_query_norm", "none")) == "dual_softmax":
        distances = _dual_softmax_rerank(
            distances, float(card.get("ranking_query_norm_tau", 0.05))
        )
    return {"distances": distances}
