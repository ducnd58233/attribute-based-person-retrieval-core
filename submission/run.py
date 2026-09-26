"""Codabench Track-2 entry: gallery attribute probs, then configured distances.

Self-contained for the eval image (torch / torchvision / numpy / Pillow).
Weights live under assets/ (written by scripts/package_submission.py).
"""

from __future__ import annotations

import json
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


class _HasFc(Protocol):
    fc: nn.Module


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
            features = self.backbone.features(images)
            pooled = self.backbone.avgpool(features)
            pooled = torch.flatten(pooled, 1)
        else:
            features = self.backbone.conv1(images)
            features = self.backbone.bn1(features)
            features = self.backbone.relu(features)
            features = self.backbone.maxpool(features)
            features = self.backbone.layer1(features)
            features = self.backbone.layer2(features)
            features = self.backbone.layer3(features)
            features = self.backbone.layer4(features)
            pooled = torch.flatten(self.backbone.avgpool(features), 1)
        return self.hard_query_adapter.encode_gallery_features(pooled)

    def query_embeddings(self, queries: torch.Tensor) -> torch.Tensor:
        return self.hard_query_adapter.encode_queries(queries)


def _feature_dim(backbone: nn.Module) -> int:
    if hasattr(backbone, "classifier"):
        head = backbone.classifier[-1]
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


def load_model() -> None:
    """Load weights once. Safe to call repeatedly."""
    if _STATE["model"] is not None:
        return
    with CARD_PATH.open(encoding="utf-8") as handle:
        card = json.load(handle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(card)
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    _STATE["model"] = model
    _STATE["device"] = device
    _STATE["card"] = card


def _iter_gallery_tensors(gallery: list[dict[str, Any]], device: torch.device) -> Any:
    total = len(gallery)
    for start in range(0, total, BATCH_SIZE):
        chunk = gallery[start : start + BATCH_SIZE]
        tensors: list[torch.Tensor] = []
        for item in chunk:
            image = Image.open(item["image_path"]).convert("RGB")
            tensor = IMAGE_TRANSFORM(image)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Image transform must return a tensor")
            tensors.append(tensor)
        images = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        done = min(start + BATCH_SIZE, total)
        print(f"[rank_gallery] {done}/{total} gallery images processed", flush=True)
        yield images


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


def _predict_gallery(gallery: list[dict[str, Any]]) -> np.ndarray:
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
    tta_scale = float(card.get("inference_tta_scale", 1.15))
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images in _iter_gallery_tensors(gallery, device):
            probs_sum: torch.Tensor | None = None
            for view in tta_views:
                logits = model(_apply_tta_view(images, view, scale=tta_scale))
                probs = torch.sigmoid(logits)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            assert probs_sum is not None
            batches.append((probs_sum / float(len(tta_views))).float().cpu().numpy())
    if not batches:
        raise ValueError("Gallery is empty")
    return np.concatenate(batches, axis=0)


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
    return probabilities.sum(1)[None, :] + queries @ (1.0 - 2.0 * probabilities).T


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
    return -(queries @ logits.T) - log_1mp.sum(axis=1)[None, :]


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
    row_min = distances.min(axis=1, keepdims=True)
    row_max = distances.max(axis=1, keepdims=True)
    spread = row_max - row_min
    safe = np.maximum(spread, 1e-8)
    normalised = (distances - row_min) / safe
    return np.where(spread > 0, normalised, np.zeros_like(distances))


def _hybrid_distance(
    queries: np.ndarray,
    probabilities: np.ndarray,
    temperature: float,
    alpha: float,
) -> np.ndarray:
    nll = _minmax_per_query(_nll_distance(queries, probabilities, temperature))
    l1 = _minmax_per_query(_l1_distance(queries, probabilities))
    weight = float(alpha)
    return weight * nll + (1.0 - weight) * l1


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


def _dual_softmax_rerank(distances: np.ndarray, tau: float) -> np.ndarray:
    """S' = S + log_softmax_queries(S/tau); return -S'. No labels / updates."""
    similarities = -distances.astype(np.float64)
    scale = max(float(tau), 1e-8)
    scaled = similarities / scale
    row_max = scaled.max(axis=0, keepdims=True)
    exp_shifted = np.exp(scaled - row_max)
    log_partition = np.log(np.maximum(exp_shifted.sum(axis=0, keepdims=True), 1e-12))
    log_softmax = scaled - row_max - log_partition
    adjusted = similarities + log_softmax
    return (-adjusted).astype(np.float32)


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
    probabilities = _predict_gallery(sample["gallery"])
    queries = np.asarray(sample["queries"], dtype=np.float32)
    card = _STATE["card"] if isinstance(_STATE["card"], dict) else {}
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
