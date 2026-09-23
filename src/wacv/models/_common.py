from __future__ import annotations

from torch import nn


def classification_head(
    in_features: int, num_attributes: int, dropout: float
) -> nn.Module:
    if dropout == 0.0:
        return nn.Linear(in_features, num_attributes)
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_attributes))
