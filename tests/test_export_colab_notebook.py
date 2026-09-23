"""Collector exporter: scripts/ + src.wacv → notebook cells."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_exporter() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "export-colab-notebook.py"
    spec = importlib.util.spec_from_file_location("export_colab_notebook", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_src_includes_engine() -> None:
    exporter = _load_exporter()
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "scripts" / "download-dataset.py",
        root / "scripts" / "train.py",
        root / "scripts" / "evaluate.py",
        root / "scripts" / "package-submission.py",
    ]
    collected = exporter.collect_src_files(scripts, root)
    rels = {path.relative_to(root).as_posix() for path in collected}
    assert "src/wacv/engine.py" in rels
    assert "src/wacv/config.py" in rels
    assert "src/wacv/models/factory.py" in rels


def test_build_cells_bakes_config() -> None:
    exporter = _load_exporter()
    root = Path(__file__).resolve().parents[1]
    config_rel = "configs/convnext_baseline.yml"
    cells = exporter.build_cells(root, config_rel=config_rel)
    notebook = exporter.cells_to_notebook(cells)
    code = exporter.notebook_code_text(notebook)
    assert "%%writefile scripts/train.py" in code
    assert "%%writefile src/wacv/engine.py" in code
    assert config_rel in code
    assert exporter.REQUIRED_FROM_TRAIN in code
    assert exporter.REQUIRED_FROM_SRC in code
    assert "runpy.run_path('scripts/train.py'" in code
    assert "subprocess.check_call" not in code
