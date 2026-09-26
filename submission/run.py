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


def _predict_gallery(gallery: list[dict[str, Any]]) -> np.ndarray:
    load_model()
    model = _STATE["model"]
    device = _STATE["device"]
    if not isinstance(model, nn.Module) or not isinstance(device, torch.device):
        raise RuntimeError("Model failed to load")
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images in _iter_gallery_tensors(gallery, device):
            logits = model(images)
            batches.append(torch.sigmoid(logits).float().cpu().numpy())
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


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    """Return distances between query vectors and gallery probabilities."""
    probabilities = _predict_gallery(sample["gallery"])
    queries = np.asarray(sample["queries"], dtype=np.float32)
    card = _STATE["card"] if isinstance(_STATE["card"], dict) else {}
    distance_name = str(card.get("ranking_distance", "l1"))
    temperature = float(card.get("ranking_temperature", 1.5))
    if distance_name == "clear_hard":
        load_model()
        model = _STATE["model"]
        device = _STATE["device"]
        if not isinstance(model, _AttributeModelWithHardQuery):
            raise TypeError("clear_hard requires hard_query_adapter")
        if not isinstance(device, torch.device):
            raise RuntimeError("Model failed to load")
        gallery_embeddings = _predict_gallery_embeddings(sample["gallery"])
        with torch.inference_mode():
            query_embeddings = (
                model.query_embeddings(
                    torch.as_tensor(queries, dtype=torch.float32, device=device)
                )
                .float()
                .cpu()
                .numpy()
            )
        distances = _clear_hard_distance(
            queries,
            probabilities,
            gallery_embeddings,
            query_embeddings,
            temperature=temperature,
            clear_fusion=float(card.get("clear_fusion", 0.5)),
        )
    elif distance_name == "specker":
        distances = _specker_distance(queries, probabilities, temperature)
    elif distance_name == "nll":
        distances = _nll_distance(queries, probabilities, temperature)
    else:
        distances = _l1_distance(queries, probabilities)
    return {"distances": distances}
