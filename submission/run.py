"""Codabench Track-2 entry: gallery attribute probs, then L1 distances.

Self-contained for the eval image (torch / torchvision / numpy / Pillow).
Weights live under assets/ (written by scripts/package_submission.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import convnext_base, convnext_tiny, resnet50

HERE = Path(__file__).resolve().parent
WEIGHTS_PATH = HERE / "assets" / "model.pt"
CARD_PATH = HERE / "assets" / "model_card.json"
BATCH_SIZE = 64

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_STATE: dict[str, Any] = {"model": None, "device": None}


class _HasFc(Protocol):
    fc: nn.Module


def _classification_head(
    in_features: int, num_attributes: int, dropout: float
) -> nn.Module:
    if dropout == 0.0:
        return nn.Linear(in_features, num_attributes)
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_attributes))


def _build_model(card: dict[str, Any]) -> nn.Module:
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


def _predict_gallery(gallery: list[dict[str, Any]]) -> np.ndarray:
    load_model()
    model = _STATE["model"]
    device = _STATE["device"]
    if not isinstance(model, nn.Module) or not isinstance(device, torch.device):
        raise RuntimeError("Model failed to load")
    total = len(gallery)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
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
            logits = model(images)
            batches.append(torch.sigmoid(logits).float().cpu().numpy())
            done = min(start + BATCH_SIZE, total)
            print(f"[rank_gallery] {done}/{total} gallery images processed", flush=True)
    if not batches:
        raise ValueError("Gallery is empty")
    return np.concatenate(batches, axis=0)


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    """Return L1 distances between query vectors and gallery probabilities."""
    probabilities = _predict_gallery(sample["gallery"])
    queries = np.asarray(sample["queries"], dtype=np.float32)
    distances = probabilities.sum(1)[None, :] + queries @ (1.0 - 2.0 * probabilities).T
    return {"distances": distances}
