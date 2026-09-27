"""WiSE-FT weight-space ensembling for CLIP attribute classifiers (E18 / H18).

Interpolates a text-initialised zero-shot head model with a fine-tuned
checkpoint: ``θ(α) = (1 − α) · θ_zs + α · θ_ft``. Alpha closer to 1 keeps more
of the fine-tuned weights (WiSE-FT, Wortsman et al., CVPR 2022).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import open_clip
import torch
from torch import nn

from src.wacv.models.clip import (
    ClipAttributeClassifier,
    build_clip_attribute_classifier,
)

logger = logging.getLogger(__name__)

_ATTR_TOKEN_RE = re.compile(r"[-_]+")


def attribute_display_phrase(attribute_name: str) -> str:
    """Turn ``Age-Young`` into a short English phrase for CLIP prompts."""
    parts = [part for part in _ATTR_TOKEN_RE.split(attribute_name) if part]
    return " ".join(parts).lower()


def positive_prompt(attribute_name: str) -> str:
    """Positive template for a binary attribute."""
    return f"a photo of a person who is {attribute_display_phrase(attribute_name)}"


def negative_prompt(attribute_name: str) -> str:
    """Negated template paired with :func:`positive_prompt`."""
    return f"a photo of a person who is not {attribute_display_phrase(attribute_name)}"


def linear_layer_from_head(head: nn.Module) -> nn.Linear:
    """Return the ``nn.Linear`` inside a plain or Dropout+Linear head."""
    if isinstance(head, nn.Linear):
        return head
    if isinstance(head, nn.Sequential):
        for module in reversed(list(head.children())):
            if isinstance(module, nn.Linear):
                return module
    raise TypeError(f"expected Linear head, got {type(head)!r}")


def text_difference_classifier_weights(
    attribute_names: Sequence[str],
    *,
    arch: str = "ViT-B-16",
    pretrained_tag: str = "openai",
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(weight, bias)`` for a 40-way head from CLIP text differences.

    Weight rows are L2-normalised ``encode(pos) − encode(neg)``; bias is zero.
    """
    if not attribute_names:
        raise ValueError("attribute_names must be non-empty")
    target = device or torch.device("cpu")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained_tag
    )
    clip_model = clip_model.to(target)
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer(arch)
    positives = [positive_prompt(name) for name in attribute_names]
    negatives = [negative_prompt(name) for name in attribute_names]
    with torch.inference_mode():
        pos_tokens = tokenizer(positives).to(target)
        neg_tokens = tokenizer(negatives).to(target)
        pos_features = clip_model.encode_text(pos_tokens)
        neg_features = clip_model.encode_text(neg_tokens)
        diff = pos_features.float() - neg_features.float()
        diff = torch.nn.functional.normalize(diff, dim=-1)
    bias = torch.zeros(len(attribute_names), dtype=diff.dtype)
    return diff.detach().cpu(), bias


def apply_text_initialised_head(
    model: ClipAttributeClassifier,
    attribute_names: Sequence[str],
    *,
    arch: str = "ViT-B-16",
    pretrained_tag: str = "openai",
    device: torch.device | None = None,
) -> None:
    """Overwrite ``model.head`` linear weights with text-difference vectors."""
    weight, bias = text_difference_classifier_weights(
        attribute_names,
        arch=arch,
        pretrained_tag=pretrained_tag,
        device=device,
    )
    linear = linear_layer_from_head(model.head)
    if linear.weight.shape != weight.shape:
        raise ValueError(
            f"head weight shape {tuple(linear.weight.shape)} != "
            f"text weight shape {tuple(weight.shape)}"
        )
    with torch.no_grad():
        linear.weight.copy_(weight.to(dtype=linear.weight.dtype))
        if linear.bias is not None:
            linear.bias.copy_(bias.to(dtype=linear.bias.dtype))


def interpolate_state_dicts(
    zero_shot: Mapping[str, torch.Tensor],
    fine_tuned: Mapping[str, torch.Tensor],
    *,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Element-wise interpolate ``(1 − α) zs + α ft`` for shared float tensors."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    shared = set(zero_shot) & set(fine_tuned)
    if not shared:
        raise ValueError("zero-shot and fine-tuned state dicts share no keys")
    missing_zs = set(fine_tuned) - set(zero_shot)
    missing_ft = set(zero_shot) - set(fine_tuned)
    if missing_zs or missing_ft:
        logger.warning(
            "WiSE-FT key mismatch: only_ft=%s only_zs=%s",
            sorted(missing_zs)[:8],
            sorted(missing_ft)[:8],
        )
    merged: dict[str, torch.Tensor] = {}
    for key in shared:
        zs_tensor = zero_shot[key]
        ft_tensor = fine_tuned[key]
        if zs_tensor.shape != ft_tensor.shape:
            raise ValueError(
                f"shape mismatch for {key}: zs={tuple(zs_tensor.shape)} "
                f"ft={tuple(ft_tensor.shape)}"
            )
        if not zs_tensor.is_floating_point():
            merged[key] = ft_tensor.clone()
            continue
        blended = (1.0 - alpha) * zs_tensor.float() + alpha * ft_tensor.float()
        merged[key] = blended.to(dtype=ft_tensor.dtype)
    return merged


def _strip_module_prefix(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state):
        return dict(state)
    return {key.removeprefix("module."): value for key, value in state.items()}


def extract_model_state(
    checkpoint_path: Path, *, prefer_ema: bool = True
) -> dict[str, torch.Tensor]:
    """Load a trainer or raw state_dict checkpoint into a flat model state."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"unrecognised checkpoint type at {checkpoint_path}")
    if prefer_ema and isinstance(payload.get("ema_model"), dict):
        ema_state = {
            key: value
            for key, value in payload["ema_model"].items()
            if key != "n_averaged"
        }
        return _strip_module_prefix(ema_state)
    if isinstance(payload.get("model"), dict):
        return _strip_module_prefix(payload["model"])
    return _strip_module_prefix(payload)


def build_zero_shot_clip_state(
    attribute_names: Sequence[str],
    *,
    dropout: float,
    arch: str = "ViT-B-16",
    pretrained_tag: str = "openai",
    image_height: int = 256,
    image_width: int = 128,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Pretrained CLIP visual + text-initialised head (zero-shot endpoint)."""
    model = build_clip_attribute_classifier(
        num_attributes=len(attribute_names),
        pretrained=True,
        dropout=dropout,
        arch=arch,
        pretrained_tag=pretrained_tag,
        image_height=image_height,
        image_width=image_width,
    )
    apply_text_initialised_head(
        model,
        attribute_names,
        arch=arch,
        pretrained_tag=pretrained_tag,
        device=device,
    )
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def write_wise_ft_checkpoint(
    *,
    fine_tuned_checkpoint: Path,
    attribute_names: Sequence[str],
    alpha: float,
    output_path: Path,
    dropout: float = 0.1,
    arch: str = "ViT-B-16",
    pretrained_tag: str = "openai",
) -> Path:
    """Build and save one interpolated checkpoint for ``alpha``."""
    fine_tuned_state = extract_model_state(fine_tuned_checkpoint)
    zero_shot_state = build_zero_shot_clip_state(
        attribute_names,
        dropout=dropout,
        arch=arch,
        pretrained_tag=pretrained_tag,
    )
    # Project FT tensors onto the zs key set via a matching architecture.
    scaffold = build_clip_attribute_classifier(
        num_attributes=len(attribute_names),
        pretrained=False,
        dropout=dropout,
        arch=arch,
        pretrained_tag=pretrained_tag,
    )
    incompatible = scaffold.load_state_dict(fine_tuned_state, strict=False)
    if incompatible.missing_keys:
        logger.warning(
            "WiSE-FT FT load missing %d keys (first: %s)",
            len(incompatible.missing_keys),
            incompatible.missing_keys[:5],
        )
    fine_tuned_aligned = {
        key: value.detach().cpu().clone()
        for key, value in scaffold.state_dict().items()
    }
    merged = interpolate_state_dicts(zero_shot_state, fine_tuned_aligned, alpha=alpha)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged, "wise_ft_alpha": float(alpha)}, output_path)
    return output_path
