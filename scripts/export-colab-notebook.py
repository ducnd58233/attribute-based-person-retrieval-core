"""Generate a Colab/Kaggle notebook by collecting scripts/ + src/wacv.

Source of truth is the pipeline scripts and the ``src.wacv`` modules they
import. This exporter AST-walks those imports, emits ``%%writefile`` cells for
collected ``src/wacv`` modules and the chosen experiment YAML, then inlines
each stage script as a code cell (body + ``main()``, not ``runpy``). After
eval it inlines ``scripts/visualize.py`` once (val split, label vs prediction).
A final persist cell syncs ``runs/<run.name>/{checkpoints,submission}`` plus
the experiment YAML to Drive (and optionally zips the same tree). The
notebook is an on-demand artifact (not committed).

Usage:
  uv run python scripts/export-colab-notebook.py \\
      --config configs/convnext_baseline.yml
  uv run python scripts/export-colab-notebook.py \\
      --config configs/convnext_baseline.yml --name convnext_baseline
  uv run python scripts/export-colab-notebook.py \\
      --config configs/convnext_baseline.yml --out notebooks/custom.ipynb
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

import yaml

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUT: Final[Path] = ROOT / "notebooks" / "train_pipeline.ipynb"
DEFAULT_CONFIG: Final[Path] = Path("configs/convnext_baseline.yml")
SRC_PREFIX: Final[str] = "src.wacv"


@dataclass(frozen=True)
class Stage:
    """One pipeline stage backed by a real script under scripts/."""

    section: str
    script: Path
    argv: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentIdentity:
    """Run + model names read from the experiment YAML at export time."""

    run_name: str
    model_name: str


def experiment_identity(config_path: Path) -> ExperimentIdentity:
    """Load ``run.name`` and ``model.name`` from an experiment YAML."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"config is not a mapping: {config_path}")
    try:
        run_name = payload["run"]["name"]
        model_name = payload["model"]["name"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(
            f"config missing run.name or model.name: {config_path}"
        ) from exc
    if not isinstance(run_name, str) or not isinstance(model_name, str):
        raise SystemExit(f"run.name and model.name must be strings: {config_path}")
    return ExperimentIdentity(run_name=run_name, model_name=model_name)


def stages_for_config(config_rel: str, *, run_name: str) -> tuple[Stage, ...]:
    """Build Download/Train/Eval/Package stages with the chosen YAML baked in."""
    best_checkpoint = f"runs/{run_name}/checkpoints/best.pt"
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
                best_checkpoint,
                "--config",
                config_rel,
            ),
        ),
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
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Full destination .ipynb path (local artifact; not committed). "
            f"Default when neither --out nor --name: {DEFAULT_OUT.as_posix()}"
        ),
    )
    out_group.add_argument(
        "--name",
        type=str,
        default=None,
        metavar="NOTEBOOK",
        help=(
            "Notebook file name under notebooks/ "
            "(adds .ipynb if missing). Example: --name convnext_baseline"
        ),
    )
    return parser.parse_args()


def resolve_notebook_out(*, out: Path | None, name: str | None) -> Path:
    """Resolve destination path from ``--out``, ``--name``, or the default."""
    if name is not None:
        filename = name if name.endswith(".ipynb") else f"{name}.ipynb"
        if Path(filename).name != filename:
            raise SystemExit(
                f"--name must be a file name, not a path: {name!r} "
                "(use --out for a full path)"
            )
        return resolve_path(Path("notebooks") / filename)
    if out is not None:
        return resolve_path(out)
    return resolve_path(DEFAULT_OUT)


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


MAIN_GUARD_RE: Final[re.Pattern[str]] = re.compile(
    r"\nif\s+__name__\s*==\s*[\"']__main__[\"']\s*:\s*\n(?:\s+main\(\)\s*\n?)+\s*$",
)

# Scripts use Path(__file__).parents[1]; notebooks have no __file__.
FILE_ROOT_RE: Final[re.Pattern[str]] = re.compile(
    r"ROOT\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[1\]"
)


def inline_stage_cell(script_text: str, argv0: str, argv: Sequence[str]) -> Cell:
    """Convert a pipeline script into a notebook cell (body + main())."""
    body = MAIN_GUARD_RE.sub("\n", script_text.rstrip() + "\n")
    body = FILE_ROOT_RE.sub("ROOT = Path.cwd()", body)
    argv_list = list(argv)
    footer = (
        "\n# --- notebook entry (converted from script; not runpy) ---\n"
        "import sys\n"
        f"sys.argv = [{argv0!r}] + {argv_list!r}\n"
        "main()\n"
    )
    return Cell(kind="code", source=body + footer)


def persist_checkpoints_cell(
    *, config_rel: str, run_name: str, model_name: str
) -> Cell:
    """Sync checkpoints + submission + config under ``runs/<run_name>/`` on Drive."""
    return Cell(
        kind="code",
        source=(
            "# Colab VM disk is ephemeral. Prefer Drive (works from Cursor/VS Code\n"
            "# Colab); browser download is a silent no-op on that path.\n"
            "#\n"
            "# Drive layout:\n"
            "#   MyDrive/wacv-upar-checkpoints/runs/<run_name>/\n"
            "#     checkpoints/   (best.pt, last.pt, ...)\n"
            "#     submission/    (build/submission)\n"
            "#     <config yaml>  (experiment config used for this run)\n"
            "import shutil\n"
            "from pathlib import Path\n"
            "\n"
            f"RUN_NAME = {run_name!r}\n"
            f"MODEL_NAME = {model_name!r}\n"
            f"CONFIG_REL = Path({config_rel!r})\n"
            "RUN_DIR = Path('runs') / RUN_NAME\n"
            "CKPT_SRC = RUN_DIR / 'checkpoints'\n"
            "SUBMISSION_SRC = Path('build/submission')\n"
            "\n"
            "print(f'run={RUN_NAME} model={MODEL_NAME} config={CONFIG_REL}')\n"
            "\n"
            "\n"
            "def _replace_copy(src: Path, dest: Path) -> None:\n"
            "    if dest.exists():\n"
            "        if dest.is_dir():\n"
            "            shutil.rmtree(dest)\n"
            "        else:\n"
            "            dest.unlink()\n"
            "    dest.parent.mkdir(parents=True, exist_ok=True)\n"
            "    if src.is_dir():\n"
            "        shutil.copytree(src, dest)\n"
            "    else:\n"
            "        shutil.copy2(src, dest)\n"
            "\n"
            "\n"
            "def sync_run_tree(base: Path) -> list[Path]:\n"
            '    """Write base/runs/<run_name>/{checkpoints,submission} + config."""\n'
            "    run_dest = base / 'runs' / RUN_NAME\n"
            "    if run_dest.exists():\n"
            "        shutil.rmtree(run_dest)\n"
            "    run_dest.mkdir(parents=True)\n"
            "    copied: list[Path] = []\n"
            "    if CKPT_SRC.is_dir():\n"
            "        _replace_copy(CKPT_SRC, run_dest / 'checkpoints')\n"
            "        copied.append(run_dest / 'checkpoints')\n"
            "    if SUBMISSION_SRC.is_dir():\n"
            "        _replace_copy(SUBMISSION_SRC, run_dest / 'submission')\n"
            "        copied.append(run_dest / 'submission')\n"
            "    if CONFIG_REL.is_file():\n"
            "        cfg_dest = run_dest / CONFIG_REL.name\n"
            "        shutil.copy2(CONFIG_REL, cfg_dest)\n"
            "        copied.append(cfg_dest)\n"
            "    return copied\n"
            "\n"
            "\n"
            "copied: list[Path] = []\n"
            "try:\n"
            "    from google.colab import drive\n"
            "\n"
            "    drive.mount('/content/drive')\n"
            "    drive_root = Path('/content/drive/MyDrive/wacv-upar-checkpoints')\n"
            "    drive_root.mkdir(parents=True, exist_ok=True)\n"
            "    copied = sync_run_tree(drive_root)\n"
            "    print('Drive root:', drive_root)\n"
            "    for path in copied:\n"
            "        print(' ', path)\n"
            "    drive.flush_and_unmount()\n"
            "except ImportError:\n"
            "    print('Not Colab; skip Drive mount.')\n"
            "\n"
            "staging = Path('wacv_artifacts')\n"
            "if staging.exists():\n"
            "    shutil.rmtree(staging)\n"
            "local_copied = sync_run_tree(staging)\n"
            "print('Local staging:', local_copied)\n"
            "\n"
            "archive = None\n"
            "if local_copied:\n"
            "    archive = Path(\n"
            "        shutil.make_archive('wacv_artifacts', 'zip', staging)\n"
            "    )\n"
            "    print('Bundled into', archive)\n"
            "\n"
            "try:\n"
            "    from google.colab import files\n"
            "\n"
            "    if archive is not None:\n"
            "        print('Downloading', archive)\n"
            "        files.download(str(archive))\n"
            "except ImportError:\n"
            "    print('Not Colab; zip (if any) remains on local disk.')\n"
            "\n"
            "if archive is not None:\n"
            "    from IPython.display import FileLink, display\n"
            "\n"
            "    print(\n"
            "        'If no browser download started '\n"
            "        '(common in Cursor/VS Code Colab), '\n"
            "        'use Drive above or this link from the web Colab UI:'\n"
            "    )\n"
            "    display(FileLink(str(archive)))\n"
        ),
    )


def visualize_stage(config_rel: str, *, split: str = "val", limit: int = 4) -> Stage:
    """Build an inlined visualize stage (default: val, post-eval)."""
    return Stage(
        "## Visualize (after eval)",
        Path("scripts/visualize.py"),
        (
            "--config",
            config_rel,
            "--split",
            split,
            "--limit",
            str(limit),
        ),
    )


def device_cells() -> list[Cell]:
    """`## Device` section: resolve CUDA → TPU → CPU once for every stage."""
    return [
        Cell(kind="markdown", source="## Device"),
        Cell(
            kind="code",
            source=(
                "# TPU only: %pip install torch_xla  # match this runtime\n"
                "from src.wacv.engine import resolve_device\n"
                "\n"
                "DEVICE = resolve_device('auto')\n"
                "print('device', DEVICE)\n"
            ),
        ),
    ]


def resolved_submission_files() -> list[Path]:
    """Locate + validate the tracked submission/run.py + metadata.yaml template."""
    files = [
        resolve_path(Path("submission") / name) for name in ("run.py", "metadata.yaml")
    ]
    for path in files:
        if not path.is_file():
            raise SystemExit(f"missing submission template file: {path}")
    return files


def submission_template_cells(
    submission_files: Sequence[Path], root: Path
) -> list[Cell]:
    """`## Submission template` section: writefile cells for run.py + metadata.yaml."""
    cells: list[Cell] = [
        Cell(
            kind="markdown",
            source=(
                "## Submission template\n\n"
                "`run.py` + `metadata.yaml` the `## Package` stage copies "
                "into `build/submission/`."
            ),
        )
    ]
    for path in submission_files:
        cells.append(
            writefile_cell(rel_posix(path, root), path.read_text(encoding="utf-8"))
        )
    return cells


def build_cells(root: Path, *, config_rel: str) -> list[Cell]:
    config_path = resolve_path(Path(config_rel))
    if not config_path.is_file():
        raise SystemExit(f"missing config: {config_path}")
    identity = experiment_identity(config_path)

    stages = stages_for_config(config_rel, run_name=identity.run_name)
    visualize = visualize_stage(config_rel, split="val", limit=4)
    scripts = [resolve_path(stage.script) for stage in stages]
    scripts.append(resolve_path(visualize.script))
    for script in scripts:
        if not script.is_file():
            raise SystemExit(f"missing pipeline script: {script}")

    src_files = collect_src_files(scripts, root)
    if not src_files:
        raise SystemExit("AST walk found no src.wacv modules — refusing empty collect")

    # scripts/package-submission.py copies these with shutil, not `import`, so
    # the AST walk above never discovers them; write them explicitly or the
    # Package stage fails with FileNotFoundError on a fresh Colab filesystem.
    submission_files = resolved_submission_files()

    writefile_paths = [rel_posix(path, root) for path in src_files]
    writefile_paths.append(config_rel)
    writefile_paths.extend(rel_posix(path, root) for path in submission_files)
    # Always POSIX: Windows Path.parent stringifies with backslashes, and Colab
    # (Linux) would then mkdir a literal "src\\wacv" name instead of src/wacv.
    parents = sorted(
        {
            Path(path).parent.as_posix()
            for path in writefile_paths
            if Path(path).parent != Path()
        }
    )
    parents_literal = repr(parents)

    cells: list[Cell] = [
        Cell(
            kind="markdown",
            source=(
                "# Track 2 Colab / Kaggle pipeline\n\n"
                "Generated by `scripts/export-colab-notebook.py` from "
                "`scripts/` + collected `src/wacv` modules "
                f"(config=`{config_rel}`). "
                "Stage cells are inlined script conversions (not runpy)."
            ),
        ),
        Cell(
            kind="markdown",
            source=(
                "## Setup\n\n"
                "Write collected `src/wacv` modules and the experiment YAML, "
                "then run each inlined stage. Device uses "
                "`resolve_device('auto')` (CUDA if available, else TPU via "
                "`torch_xla` when present, else CPU)."
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
                f"for parent in {parents_literal}:\n"
                "    Path(parent).mkdir(parents=True, exist_ok=True)\n"
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

    cells.append(Cell(kind="markdown", source="## Experiment config"))
    cells.append(writefile_cell(config_rel, config_path.read_text(encoding="utf-8")))
    cells.extend(submission_template_cells(submission_files, root))
    cells.extend(device_cells())

    for stage in stages:
        script = resolve_path(stage.script)
        rel = rel_posix(script, root)
        cells.append(Cell(kind="markdown", source=stage.section))
        cells.append(
            inline_stage_cell(script.read_text(encoding="utf-8"), rel, stage.argv)
        )
        if stage.section == "## Eval":
            viz_script = resolve_path(visualize.script)
            cells.append(Cell(kind="markdown", source=visualize.section))
            cells.append(
                inline_stage_cell(
                    viz_script.read_text(encoding="utf-8"),
                    rel_posix(viz_script, root),
                    visualize.argv,
                )
            )

    cells.append(Cell(kind="markdown", source="## Persist checkpoints"))
    cells.append(
        persist_checkpoints_cell(
            config_rel=config_rel,
            run_name=identity.run_name,
            model_name=identity.model_name,
        )
    )
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


def config_relpath(config: Path, root: Path) -> str:
    resolved = resolve_path(config)
    if not resolved.is_file():
        raise SystemExit(f"missing config: {resolved}")
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


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
    out = resolve_notebook_out(out=args.out, name=args.name)
    config_rel = config_relpath(args.config, ROOT)
    write_notebook(out, config_rel=config_rel)


if __name__ == "__main__":
    main()
