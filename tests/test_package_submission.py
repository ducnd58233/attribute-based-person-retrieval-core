"""Zip-step tests for scripts/package-submission.py (no checkpoint/model needed)."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "package-submission.py"


def _load_package_submission() -> ModuleType:
    spec = importlib.util.spec_from_file_location("package_submission", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_fake_build(source: Path) -> None:
    (source / "assets").mkdir(parents=True)
    (source / "run.py").write_text("print('entry point')\n", encoding="utf-8")
    (source / "metadata.yaml").write_text("command: noop\n", encoding="utf-8")
    (source / "assets" / "model.pt").write_bytes(b"fake-weights")
    (source / "assets" / "model_card.json").write_text("{}", encoding="utf-8")
    pycache = source / "assets" / "__pycache__"
    pycache.mkdir()
    (pycache / "model.cpython-311.pyc").write_bytes(b"junk")
    (source / ".DS_Store").write_bytes(b"junk")


def test_zip_submission_has_run_py_at_root(tmp_path: Path) -> None:
    module = _load_package_submission()
    source = tmp_path / "submission"
    _make_fake_build(source)
    destination = tmp_path / "submission.zip"

    module._zip_submission(source, destination)

    with zipfile.ZipFile(destination) as archive:
        names = set(archive.namelist())
    assert "run.py" in names
    assert "metadata.yaml" in names
    assert "assets/model.pt" in names
    assert "assets/model_card.json" in names


def test_zip_submission_excludes_pycache_and_ds_store(tmp_path: Path) -> None:
    module = _load_package_submission()
    source = tmp_path / "submission"
    _make_fake_build(source)
    destination = tmp_path / "submission.zip"

    module._zip_submission(source, destination)

    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
    assert not any("__pycache__" in name for name in names)
    assert not any(name.endswith(".DS_Store") for name in names)
    assert not any(name.startswith(".git/") for name in names)


def test_zip_submission_no_wrapping_subfolder(tmp_path: Path) -> None:
    """run.py must be a top-level entry, never nested under e.g. 'submission/run.py'."""
    module = _load_package_submission()
    source = tmp_path / "submission"
    _make_fake_build(source)
    destination = tmp_path / "submission.zip"

    module._zip_submission(source, destination)

    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
    assert all("\\" not in name for name in names)
    assert all(not name.startswith("submission/") for name in names)
