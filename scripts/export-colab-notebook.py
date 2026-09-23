"""Generate a Colab/Kaggle notebook by collecting scripts/ + src/wacv.

Source of truth is the four pipeline scripts and the ``src.wacv`` modules they
import. This exporter AST-walks those imports, emits ``%%writefile`` cells for
every collected path, then stage cells that ``runpy`` each script. The notebook
is an on-demand artifact (not committed). There is no ``train_pipeline.py``.

Usage:
  uv run python scripts/export-colab-notebook.py \\
      --config configs/convnext_baseline.yml
  uv run python scripts/export-colab-notebook.py --check \\
      --config configs/convnext_baseline.yml
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUT: Final[Path] = ROOT / "notebooks" / "train_pipeline.ipynb"
DEFAULT_CONFIG: Final[Path] = Path("configs/convnext_baseline.yml")
LEGACY_PIPELINE_PY: Final[Path] = ROOT / "notebooks" / "train_pipeline.py"
LEGACY_EXPORTER: Final[Path] = ROOT / "scripts" / "export_colab_notebook.py"
SRC_PREFIX: Final[str] = "src.wacv"


@dataclass(frozen=True)
class Stage:
    """One pipeline stage backed by a real script under scripts/."""

    section: str
    script: Path
    argv: tuple[str, ...]


def stages_for_config(config_rel: str) -> tuple[Stage, ...]:
    """Build Download/Train/Eval/Package stages with the chosen YAML baked in."""
    return (
        Stage("## Download", Path("scripts/download-dataset.py"), ()),
        Stage(
            "## Train",
            Path("scripts/train.py"),
            ("--config", config_rel),
        ),
        Stage(
            "## Eval",
            Path("scripts/evaluate.py"),
            ("--config", config_rel),
        ),
        Stage(
            "## Package",
            Path("scripts/package-submission.py"),
            (
                "--checkpoint",
                "runs/convnext-baseline/checkpoints/best.pt",
                "--config",
                config_rel,
            ),
        ),
    )


REQUIRED_SECTIONS: Final[tuple[str, ...]] = (
    "## Download",
    "## Train",
    "## Eval",
    "## Package",
)
REQUIRED_FROM_TRAIN: Final[str] = "def build_loaders("
REQUIRED_FROM_SRC: Final[str] = "class Trainer"
FORBIDDEN_SUBPROCESS: Final[re.Pattern[str]] = re.compile(
    r"subprocess\.(?:check_call|run|Popen|call)\s*\([^)]*"
    r"scripts[/\\](?:train|evaluate|package-submission|download-dataset)\.py",
    re.IGNORECASE | re.DOTALL,
)

NBFORMAT_MAJOR: Final[int] = 4
NBFORMAT_MINOR: Final[int] = 5

CellKind = Literal["code", "markdown"]


@dataclass(frozen=True)
class Cell:
    """One notebook cell."""

    kind: CellKind
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Experiment YAML baked into train/eval/package cells",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Destination .ipynb path (local artifact; not committed)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="After write, assert structural acceptance",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _is_under_src_wacv(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return relative.as_posix().startswith("src/wacv/")


def _module_to_file(module: str, root: Path) -> Path | None:
    """Map ``src.wacv.foo`` to a file under root, if it exists."""
    if module != SRC_PREFIX and not module.startswith(SRC_PREFIX + "."):
        return None
    parts = module.split(".")
    base = root.joinpath(*parts)
    as_file = base.with_suffix(".py")
    as_pkg = base / "__init__.py"
    if as_file.is_file():
        return as_file.resolve()
    if as_pkg.is_file():
        return as_pkg.resolve()
    return None


def _imported_src_modules(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == SRC_PREFIX or alias.name.startswith(SRC_PREFIX + "."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            module = node.module
            if module == SRC_PREFIX or module.startswith(SRC_PREFIX + "."):
                found.add(module)
                for alias in node.names:
                    if alias.name != "*":
                        found.add(f"{module}.{alias.name}")
    return found


def _resolve_import_target(module: str, root: Path) -> Path | None:
    target = _module_to_file(module, root)
    if target is not None:
        return target
    if "." not in module:
        return None
    return _module_to_file(module.rsplit(".", 1)[0], root)


def collect_src_files(entry_scripts: Sequence[Path], root: Path) -> list[Path]:
    """Return ``src.wacv`` files reachable from entry scripts, dependency-first."""
    pending: list[Path] = [path.resolve() for path in entry_scripts]
    seen: set[Path] = set()
    src_files: set[Path] = set()

    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"missing source file: {path}")
        seen.add(path)
        if _is_under_src_wacv(path, root):
            src_files.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in sorted(_imported_src_modules(tree)):
            target = _resolve_import_target(module, root)
            if target is not None and target not in seen:
                pending.append(target)

    return _topo_src(src_files, root)


def _topo_src(files: Iterable[Path], root: Path) -> list[Path]:
    file_set = {path.resolve() for path in files}
    deps: dict[Path, set[Path]] = {path: set() for path in file_set}
    for path in file_set:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _imported_src_modules(tree):
            target = _resolve_import_target(module, root)
            if target is not None and target in file_set and target != path:
                deps[path].add(target)

    ordered: list[Path] = []
    visiting: set[Path] = set()
    visited: set[Path] = set()

    def visit(node: Path) -> None:
        if node in visited or node in visiting:
            return
        visiting.add(node)
        for dep in sorted(deps[node], key=lambda item: item.as_posix()):
            visit(dep)
        visiting.remove(node)
        visited.add(node)
        ordered.append(node)

    for path in sorted(file_set, key=lambda item: item.as_posix()):
        visit(path)
    return ordered


def rel_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def writefile_cell(relpath: str, content: str) -> Cell:
    body = content if content.endswith("\n") else f"{content}\n"
    return Cell(kind="code", source=f"%%writefile {relpath}\n{body}")


def run_stage_cell(relpath: str, argv: Sequence[str]) -> Cell:
    argv_list = list(argv)
    source = (
        "import runpy\n"
        "import sys\n"
        f"sys.argv = [{relpath!r}] + {argv_list!r}\n"
        f"runpy.run_path({relpath!r}, run_name='__main__')\n"
    )
    return Cell(kind="code", source=source)


def build_cells(root: Path, *, config_rel: str) -> list[Cell]:
    stages = stages_for_config(config_rel)
    scripts = [resolve_path(stage.script) for stage in stages]
    for script in scripts:
        if not script.is_file():
            raise SystemExit(f"missing pipeline script: {script}")

    src_files = collect_src_files(scripts, root)
    if not src_files:
        raise SystemExit("AST walk found no src.wacv modules — refusing empty collect")

    cells: list[Cell] = [
        Cell(
            kind="markdown",
            source=(
                "# Track 2 Colab / Kaggle pipeline\n\n"
                "Generated by `scripts/export-colab-notebook.py` from "
                "`scripts/` + collected `src/wacv` modules "
                f"(config=`{config_rel}`). "
                "Do not hand-edit; regenerate the notebook instead."
            ),
        ),
        Cell(
            kind="markdown",
            source=(
                "## Setup\n\n"
                "Write collected library modules, then run each stage script."
            ),
        ),
        Cell(
            kind="code",
            source=(
                "from pathlib import Path\n"
                "import sys\n"
                "ROOT = Path.cwd()\n"
                "if str(ROOT) not in sys.path:\n"
                "    sys.path.insert(0, str(ROOT))\n"
                "print('ROOT', ROOT)\n"
            ),
        ),
        Cell(
            kind="markdown",
            source="## Collected library (`src/wacv`)",
        ),
    ]

    for path in src_files:
        cells.append(
            writefile_cell(rel_posix(path, root), path.read_text(encoding="utf-8"))
        )

    for stage in stages:
        script = resolve_path(stage.script)
        rel = rel_posix(script, root)
        cells.append(Cell(kind="markdown", source=stage.section))
        cells.append(writefile_cell(rel, script.read_text(encoding="utf-8")))
        cells.append(run_stage_cell(rel, stage.argv))

    return cells


def _source_lines(text: str) -> list[str]:
    body = text.strip("\n")
    if not body:
        return []
    parts = body.split("\n")
    lines = [f"{line}\n" for line in parts[:-1]]
    lines.append(parts[-1])
    return lines


def cells_to_notebook(cells: list[Cell]) -> dict[str, object]:
    notebook_cells: list[dict[str, object]] = []
    for index, cell in enumerate(cells):
        if cell.kind == "markdown":
            notebook_cells.append(
                {
                    "cell_type": "markdown",
                    "id": str(index),
                    "metadata": {},
                    "source": _source_lines(cell.source),
                }
            )
        else:
            notebook_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": str(index),
                    "metadata": {},
                    "outputs": [],
                    "source": _source_lines(cell.source),
                }
            )
    return {
        "nbformat": NBFORMAT_MAJOR,
        "nbformat_minor": NBFORMAT_MINOR,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": notebook_cells,
    }


def _flatten_source(source: object) -> str:
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    if isinstance(source, str):
        return source
    return ""


def notebook_text(notebook: dict[str, object]) -> str:
    chunks: list[str] = []
    raw_cells = notebook.get("cells")
    if not isinstance(raw_cells, list):
        return ""
    for cell in raw_cells:
        if isinstance(cell, dict):
            chunks.append(_flatten_source(cell.get("source", [])))
    return "".join(chunks)


def notebook_code_text(notebook: dict[str, object]) -> str:
    chunks: list[str] = []
    raw_cells = notebook.get("cells")
    if not isinstance(raw_cells, list):
        return ""
    for cell in raw_cells:
        if isinstance(cell, dict) and cell.get("cell_type") == "code":
            chunks.append(_flatten_source(cell.get("source", [])))
    return "".join(chunks)


def config_relpath(config: Path, root: Path) -> str:
    resolved = resolve_path(config)
    if not resolved.is_file():
        raise SystemExit(f"missing config: {resolved}")
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def validate_notebook(
    path: Path, *, root: Path = ROOT, config_rel: str = DEFAULT_CONFIG.as_posix()
) -> None:
    if LEGACY_PIPELINE_PY.is_file():
        raise SystemExit(
            f"legacy duplicate still present: {LEGACY_PIPELINE_PY} "
            "(delete it; notebook is generated from scripts/ + src/)"
        )
    if LEGACY_EXPORTER.is_file():
        raise SystemExit(
            f"legacy exporter still present: {LEGACY_EXPORTER} "
            "(use scripts/export-colab-notebook.py)"
        )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != NBFORMAT_MAJOR:
        raise SystemExit(
            f"expected nbformat {NBFORMAT_MAJOR}, got {notebook.get('nbformat')}"
        )
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise SystemExit("notebook has no cells")
    text = notebook_text(notebook)
    code = notebook_code_text(notebook)
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in text]
    if missing_sections:
        raise SystemExit(f"missing markdown sections: {missing_sections}")
    if REQUIRED_FROM_TRAIN not in code:
        raise SystemExit(
            f"notebook missing collected train.py marker: {REQUIRED_FROM_TRAIN}"
        )
    if REQUIRED_FROM_SRC not in code:
        raise SystemExit(
            f"notebook missing collected src/wacv marker: {REQUIRED_FROM_SRC}"
        )
    if "smoke.yml" in code:
        raise SystemExit("pipeline notebook must not reference smoke.yml")
    if config_rel not in code:
        raise SystemExit(f"pipeline notebook must reference --config {config_rel}")
    if FORBIDDEN_SUBPROCESS.search(code):
        raise SystemExit(
            "notebook still shells out to pipeline scripts via subprocess "
            "(use collected writefile + runpy instead)"
        )
    if "%%writefile scripts/train.py" not in code:
        raise SystemExit("notebook missing writefile for scripts/train.py")
    if "%%writefile src/wacv/engine.py" not in code:
        raise SystemExit("notebook missing writefile for src/wacv/engine.py")
    print(f"check ok: {path} ({len(cells)} cells, root={root})")


def write_notebook(out: Path, *, root: Path = ROOT, config_rel: str) -> None:
    cells = build_cells(root, config_rel=config_rel)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(cells_to_notebook(cells), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out} ({len(cells)} cells) from scripts/ + src/wacv")


def main() -> None:
    args = parse_args()
    out = resolve_path(args.out)
    config_rel = config_relpath(args.config, ROOT)
    if LEGACY_PIPELINE_PY.is_file():
        LEGACY_PIPELINE_PY.unlink()
        print(f"removed legacy duplicate {LEGACY_PIPELINE_PY}")
    if LEGACY_EXPORTER.is_file():
        LEGACY_EXPORTER.unlink()
        print(f"removed legacy exporter {LEGACY_EXPORTER}")
    write_notebook(out, config_rel=config_rel)
    validate_notebook(out, config_rel=config_rel)
    if args.check:
        print("--check passed")


if __name__ == "__main__":
    main()
