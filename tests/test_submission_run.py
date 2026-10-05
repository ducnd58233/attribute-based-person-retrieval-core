"""Progress-logging tests for submission/run.py's rank_gallery path.

Loaded by file path (not as a package import) because submission/run.py must
stay self-contained for the eval image and is never imported from src.wacv.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
RUN_PY_PATH = ROOT / "submission" / "run.py"


def _load_run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("submission_run", RUN_PY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.shape[0], 3)


def _make_gallery(tmp_path: Path, count: int) -> list[dict[str, object]]:
    gallery = []
    for index in range(count):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (8, 8), color=(index, index, index)).save(path)
        gallery.append({"index": index, "image": str(path), "image_path": str(path)})
    return gallery


def test_predict_gallery_prints_progress_per_batch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_run_module()
    module._STATE["model"] = _FakeModel()
    module._STATE["device"] = torch.device("cpu")
    module._STATE["card"] = {}
    monkeypatch.setattr(module, "BATCH_SIZE", 2)
    gallery = _make_gallery(tmp_path, count=5)

    probabilities, features = module._predict_gallery(gallery)

    assert probabilities.shape == (5, 3)
    assert features is None
    output = capsys.readouterr().out
    progress_lines = [line for line in output.splitlines() if "5" in line]
    assert len(progress_lines) >= 3  # 3 batches of size 2, 2, 1
    assert "5/5" in output


def test_predict_gallery_progress_mentions_total(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_run_module()
    module._STATE["model"] = _FakeModel()
    module._STATE["device"] = torch.device("cpu")
    module._STATE["card"] = {}
    monkeypatch.setattr(module, "BATCH_SIZE", 3)
    gallery = _make_gallery(tmp_path, count=3)

    probabilities, features = module._predict_gallery(gallery)

    assert probabilities.shape == (3, 3)
    assert features is None
    output = capsys.readouterr().out
    assert "3/3" in output


class _FakeFeatureModel(nn.Module):
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = images.shape[0]
        logits = torch.zeros(batch, 3)
        features = torch.ones(batch, 4)
        return logits, features


def test_predict_gallery_returns_identity_features_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_run_module()
    module._STATE["model"] = _FakeFeatureModel()
    module._STATE["device"] = torch.device("cpu")
    module._STATE["card"] = {"inference_tta": ["identity"]}
    monkeypatch.setattr(module, "BATCH_SIZE", 2)
    gallery = _make_gallery(tmp_path, count=3)

    probabilities, features = module._predict_gallery(gallery, with_features=True)

    assert probabilities.shape == (3, 3)
    assert features is not None
    assert features.shape == (3, 4)


def test_neighbour_smooth_mirrors_library_blend() -> None:
    module = _load_run_module()
    probabilities = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    features = np.array([[1.0, 0.0], [0.99, 0.1]], dtype=np.float32)
    smooth = {"k": 1, "beta": 1.0, "min_similarity": -1.0}

    smoothed = module._neighbour_smooth(probabilities, features, smooth)

    np.testing.assert_allclose(smoothed[0], probabilities[1], atol=1e-5)
    np.testing.assert_allclose(smoothed[1], probabilities[0], atol=1e-5)
