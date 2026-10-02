from __future__ import annotations

from collections.abc import Iterable

from torch import nn


def classification_head(
    in_features: int, num_attributes: int, dropout: float
) -> nn.Module:
    if dropout == 0.0:
        return nn.Linear(in_features, num_attributes)
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_attributes))


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Set ``requires_grad`` on every parameter of ``module``."""
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def build_layerwise_param_groups(
    *,
    head: nn.Module,
    tower: nn.Module,
    blocks: Iterable[nn.Module],
    learning_rate: float,
    weight_decay: float,
    layer_decay: float,
) -> list[dict[str, object]]:
    """AdamW param groups with deeper ViT blocks keeping higher LR.

    The head trains at ``learning_rate``; tower parameters outside ``blocks``
    (patch/positional embeddings) get ``layer_decay ** num_layers``; block
    ``i`` gets ``layer_decay ** (num_layers - 1 - i)``. Frozen parameters are
    skipped.
    """
    block_list = list(blocks)
    num_layers = len(block_list)
    layer_param_ids = {
        id(parameter) for block in block_list for parameter in block.parameters()
    }
    groups: list[dict[str, object]] = []
    head_params = [p for p in head.parameters() if p.requires_grad]
    if head_params:
        groups.append(
            {"params": head_params, "lr": learning_rate, "weight_decay": weight_decay}
        )
    early_params = [
        parameter
        for parameter in tower.parameters()
        if parameter.requires_grad and id(parameter) not in layer_param_ids
    ]
    if early_params:
        groups.append(
            {
                "params": early_params,
                "lr": learning_rate * layer_decay**num_layers,
                "weight_decay": weight_decay,
            }
        )
    for index, block in enumerate(block_list):
        params = [p for p in block.parameters() if p.requires_grad]
        if params:
            groups.append(
                {
                    "params": params,
                    "lr": learning_rate * layer_decay ** (num_layers - 1 - index),
                    "weight_decay": weight_decay,
                }
            )
    return groups
