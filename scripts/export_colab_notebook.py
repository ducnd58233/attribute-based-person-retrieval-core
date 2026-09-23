"""Convert notebooks/train_pipeline.py (# %% cells) into train_pipeline.ipynb.

The pipeline source holds the real download/train/eval/package orchestration.
This script is a generic cell splitter — it does not hardcode cell bodies.

Usage:
  uv run python scripts/export_colab_notebook.py
  uv run python scripts/export_colab_notebook.py --check
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE: Final[Path] = ROOT / "notebooks" / "train_pipeline.py"
DEFAULT_OUT: Final[Path] = ROOT / "notebooks" / "train_pipeline.ipynb"

CELL_RE: Final[re.Pattern[str]] = re.compile(
    r"^# %%(?:\s*\[(?P<kind>markdown|md)\])?\s*$",
    re.MULTILINE,
)

FORBIDDEN_SCRIPT_CALLS: Final[tuple[str, ...]] = (
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/package-submission.py",
    "scripts/download-dataset.py",
    "scripts\\train.py",
    "scripts\\evaluate.py",
    "scripts\\package-submission.py",
    "scripts\\download-dataset.py",
)

REQUIRED_SECTIONS: Final[tuple[str, ...]] = (
    "## Download",
    "## Train",
    "## Eval",
    "## Package",
)

REQUIRED_LOGIC: Final[tuple[str, ...]] = (
    "Trainer(",
    "evaluate_ranking",
    "export_weights",
    "clone_starter",
)

NBFORMAT_MAJOR: Final[int] = 4
NBFORMAT_MINOR: Final[int] = 5

CellKind = Literal["code", "markdown"]


@dataclass(frozen=True)
class Cell:
    """One notebook cell parsed from a # %% source file."""

    kind: CellKind
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Cell-marked Python source",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Destination .ipynb path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="After write, assert structural acceptance",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def markdown_from_percent(body: str) -> str:
    """Turn jupytext-style commented markdown into raw markdown text."""
    lines: list[str] = []
    for line in body.split("\n"):
        if line.startswith("# "):
            lines.append(line[2:])
        elif line == "#":
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines).strip("\n")


def parse_percent_script(text: str) -> list[Cell]:
    """Split a jupytext-style percent script into cells."""
    matches = list(CELL_RE.finditer(text))
    if not matches:
        raise SystemExit("no # %% cell markers found in source")

    cells: list[Cell] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip("\n")
        kind: CellKind = (
            "markdown" if match.group("kind") in {"markdown", "md"} else "code"
        )
        if kind == "markdown":
            body = markdown_from_percent(body)
        cells.append(Cell(kind=kind, source=body))
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
    """Flatten code cells only (markdown may name scripts without invoking them)."""
    chunks: list[str] = []
    raw_cells = notebook.get("cells")
    if not isinstance(raw_cells, list):
        return ""
    for cell in raw_cells:
        if isinstance(cell, dict) and cell.get("cell_type") == "code":
            chunks.append(_flatten_source(cell.get("source", [])))
    return "".join(chunks)


def validate_notebook(path: Path) -> None:
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
    missing_logic = [token for token in REQUIRED_LOGIC if token not in code]
    if missing_logic:
        raise SystemExit(f"notebook missing inlined logic markers: {missing_logic}")
    if "smoke.yml" in code:
        raise SystemExit("pipeline notebook must not reference smoke.yml")
    if "convnext_baseline.yml" not in code:
        raise SystemExit("pipeline notebook must reference convnext_baseline.yml")
    hits = [name for name in FORBIDDEN_SCRIPT_CALLS if name in code]
    if hits:
        raise SystemExit(
            f"notebook still shells out to pipeline scripts (use inlined logic): {hits}"
        )
    print(f"check ok: {path} ({len(cells)} cells)")


def write_notebook(source: Path, out: Path) -> None:
    cells = parse_percent_script(source.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(cells_to_notebook(cells), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out} from {source} ({len(cells)} cells)")


def main() -> None:
    args = parse_args()
    source = resolve_path(args.source)
    out = resolve_path(args.out)
    if not source.is_file():
        raise SystemExit(f"missing source: {source}")
    write_notebook(source, out)
    validate_notebook(out)
    if args.check:
        print("--check passed")


if __name__ == "__main__":
    main()
