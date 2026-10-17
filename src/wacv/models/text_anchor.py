"""Initialise a CLIP attribute head from text embeddings of prompts (E77 / H80)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, cast

import open_clip
import torch
import torch.nn.functional as F
from torch import nn

from src.wacv.models.clip import ClipAttributeClassifier

LOGGER = logging.getLogger(__name__)

_CLIP_ARCH_BY_MODEL: dict[str, str] = {
    "clip_vit_b16": "ViT-B-16",
    "clip_vit_l14": "ViT-L-14",
}


def _linear_head(head: nn.Module) -> nn.Linear:
    if isinstance(head, nn.Linear):
        return head
    if isinstance(head, nn.Sequential):
        for module in reversed(list(head.children())):
            if isinstance(module, nn.Linear):
                return module
    raise TypeError(f"expected Linear or Sequential+Linear head, got {type(head)}")


def _prompt_pair(attribute_name: str) -> tuple[str, str]:
    phrase = attribute_name.replace("-", " ").replace("&", " and ").lower()
    positive = f"a photo of a person who is {phrase}"
    negative = f"a photo of a person who is not {phrase}"
    return positive, negative


def init_clip_head_from_text(
    classifier: ClipAttributeClassifier,
    attribute_names: Sequence[str],
    *,
    model_name: str = "clip_vit_b16",
    device: torch.device | None = None,
) -> None:
    """Set the linear head from L2-normalised (pos − neg) CLIP text embeddings.

    Keeps the already-built vision tower; loads a fresh open_clip model only to
    encode text, then discards it. Bias is zeroed.
    """
    arch = _CLIP_ARCH_BY_MODEL.get(model_name)
    if arch is None:
        raise ValueError(f"text_anchor_init unsupported for model {model_name!r}")
    head = _linear_head(classifier.head)
    if head.out_features != len(attribute_names):
        raise ValueError(
            f"head out_features={head.out_features} != "
            f"len(attribute_names)={len(attribute_names)}"
        )
    if head.in_features != classifier.feature_dim:
        raise ValueError(
            f"head in_features={head.in_features} != "
            f"feature_dim={classifier.feature_dim}"
        )

    positives = [_prompt_pair(name)[0] for name in attribute_names]
    negatives = [_prompt_pair(name)[1] for name in attribute_names]
    created, _, _ = open_clip.create_model_and_transforms(arch, pretrained="openai")
    tokenizer = open_clip.get_tokenizer(arch)
    text_device = device or next(classifier.parameters()).device
    # open_clip models are dynamically typed; encode_text and logit_scale are
    # not in the nn.Module stubs, so the checker sees them as Tensors.
    text_model = cast(Any, created.to(text_device))
    text_model.eval()
    with torch.inference_mode():
        pos = text_model.encode_text(tokenizer(positives).to(text_device))
        neg = text_model.encode_text(tokenizer(negatives).to(text_device))
        delta = F.normalize(pos.float() - neg.float(), dim=-1)
        if delta.shape[-1] != head.in_features:
            raise ValueError(
                f"text dim {delta.shape[-1]} != head in_features {head.in_features}"
            )
        scale = float(text_model.logit_scale.exp().clamp(max=100.0).item())
        head.weight.copy_(delta.to(dtype=head.weight.dtype) * scale)
        if head.bias is not None:
            head.bias.zero_()
    del text_model
    LOGGER.info(
        "text-anchored head init: %d attributes, scale=%.3f, arch=%s",
        len(attribute_names),
        scale,
        arch,
    )
