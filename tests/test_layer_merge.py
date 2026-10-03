"""Unit tests for E75 layer-group merge helpers."""

from __future__ import annotations

import torch

from src.wacv.layer_merge import (
    LAYER_GROUPS,
    decode_cma_vector,
    layer_group_for_key,
    merge_by_layer_groups,
    softmax_rows,
    uniform_logits,
)


def test_layer_group_for_key_covers_clip_shapes() -> None:
    assert layer_group_for_key("visual.conv1.weight") == "stem"
    key_early = "visual.transformer.resblocks.2.attn.in_proj_weight"
    assert layer_group_for_key(key_early) == "blocks_0_3"
    key_mid = "visual.transformer.resblocks.5.mlp.c_fc.weight"
    assert layer_group_for_key(key_mid) == "blocks_4_7"
    key_late = "visual.transformer.resblocks.10.ln_1.weight"
    assert layer_group_for_key(key_late) == "blocks_8_11"
    assert layer_group_for_key("visual.ln_post.weight") == "ln_post_proj"
    assert layer_group_for_key("visual.proj") == "ln_post_proj"
    assert layer_group_for_key("classifier.1.weight") == "head"


def test_uniform_merge_matches_mean() -> None:
    states = [
        {
            "visual.conv1.weight": torch.ones(2, 2),
            "classifier.1.bias": torch.tensor([1.0, 2.0]),
        },
        {
            "visual.conv1.weight": torch.ones(2, 2) * 3,
            "classifier.1.bias": torch.tensor([3.0, 4.0]),
        },
    ]
    result = merge_by_layer_groups(
        states,
        ("a", "b"),
        uniform_logits(len(LAYER_GROUPS), 2),
    )
    expected_conv = torch.ones(2, 2) * 2
    assert torch.allclose(result.state_dict["visual.conv1.weight"], expected_conv)
    expected_bias = torch.tensor([2.0, 3.0])
    assert torch.allclose(result.state_dict["classifier.1.bias"], expected_bias)
    for row in result.coefficients:
        assert abs(sum(row) - 1.0) < 1e-5


def test_decode_and_softmax_shape() -> None:
    vector = [0.0] * (len(LAYER_GROUPS) * 3)
    logits = decode_cma_vector(vector, n_groups=len(LAYER_GROUPS), n_members=3)
    probs = softmax_rows(logits)
    assert probs.shape == (len(LAYER_GROUPS), 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(len(LAYER_GROUPS)))
