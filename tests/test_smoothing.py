"""Unit tests for latent kNN neighbour smoothing (E74 / E74b)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from src.wacv.config import NeighbourSmoothConfig, load_config
from src.wacv.evaluation import neighbour_smooth_probabilities

ROOT = Path(__file__).resolve().parents[1]


def _load_package_submission() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "package_submission", ROOT / "scripts" / "package-submission.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_neighbour_smooth_blends_identical_neighbours() -> None:
    probabilities = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    # Unit vectors on a line so 0 is nearest to 1 and 2 is farther.
    embeddings = np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]], dtype=np.float32)
    config = NeighbourSmoothConfig(k=1, beta=1.0, min_similarity=-1.0)

    smoothed = neighbour_smooth_probabilities(probabilities, embeddings, config)

    np.testing.assert_allclose(smoothed[0], probabilities[1], atol=1e-5)
    np.testing.assert_allclose(smoothed[1], probabilities[0], atol=1e-5)


def test_neighbour_smooth_gate_keeps_own_probs_when_none_pass() -> None:
    probabilities = np.eye(3, dtype=np.float32)
    embeddings = np.eye(3, dtype=np.float32)
    config = NeighbourSmoothConfig(k=2, beta=0.7, min_similarity=0.99)

    smoothed = neighbour_smooth_probabilities(probabilities, embeddings, config)

    np.testing.assert_allclose(smoothed, probabilities)


def test_neighbour_smooth_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="probability rows"):
        neighbour_smooth_probabilities(
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((3, 4), dtype=np.float32),
            NeighbourSmoothConfig(),
        )


def test_package_resolves_default_smoothing_for_clip() -> None:
    module = _load_package_submission()
    config = load_config(ROOT / "configs" / "e17_clip_b16_lpft.yml")

    resolved = module._resolve_neighbour_smooth(config, disable=False)

    assert resolved == {"k": 5, "beta": 0.7, "min_similarity": 0.95}


def test_package_disables_smoothing_with_flag() -> None:
    module = _load_package_submission()
    config = load_config(ROOT / "configs" / "e74b_gated_smoothing.yml")

    assert module._resolve_neighbour_smooth(config, disable=True) is None


def test_package_leaves_convnext_without_smoothing() -> None:
    module = _load_package_submission()
    config = load_config(ROOT / "configs" / "convnext_baseline.yml")

    assert module._resolve_neighbour_smooth(config, disable=False) is None
