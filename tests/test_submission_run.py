"""Progress-logging tests for submission/run.py's rank_gallery path.

Loaded by file path (not as a package import) because submission/run.py must
stay self-contained for the eval image and is never imported from src.wacv.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

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

    module._predict_gallery(gallery)

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

    module._predict_gallery(gallery)

    output = capsys.readouterr().out
    assert "3/3" in output


def test_rank_gallery_logs_device_and_phase_seconds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_run_module()
    module._STATE["model"] = _FakeModel()
    module._STATE["device"] = torch.device("cpu")
    module._STATE["card"] = {
        "ranking_distance": "l1",
        "ranking_temperature": 1.5,
        "inference_tta": ["identity"],
    }
    monkeypatch.setattr(module, "BATCH_SIZE", 2)
    gallery = _make_gallery(tmp_path, count=4)
    sample = {
        "gallery": gallery,
        "queries": [[1, 0, 0], [0, 1, 0]],
    }

    result = module.rank_gallery(sample)

    assert "distances" in result
    output = capsys.readouterr().out
    assert "phase_seconds" in output
    assert "decode=" in output
    assert "forward=" in output
    assert "scoring=" in output
    assert "image_path" not in output
    assert ".jpg" not in output
