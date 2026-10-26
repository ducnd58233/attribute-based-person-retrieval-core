"""Locate the organizer dataset, shared across git worktrees."""

from __future__ import annotations

import os
from pathlib import Path

# Shared across git worktrees; see AGENTS.md "Experiment worktrees: shared data".
DATA_ROOT_ENV = "WACV_DATA_ROOT"
MAIN_CHECKOUT_DIRNAME = "attribute-based-person-retrieval-core"

# Accepted directory names per corpus (extracted archives differ by tool).
_CORPUS_DIR_NAMES: tuple[tuple[str, ...], ...] = (
    ("Market1501", "Market-1501-v15.09.15"),
    ("PA100k", "PA100K"),
    ("PETA",),
)


def dataset_looks_present(data: Path) -> bool:
    """True when annotations and all three organizer image corpora exist.

    Requires Market1501, PA100k, and PETA (under any accepted folder name), not
    just one corpus, so a partial extract cannot skip the rest of the download.
    """
    if not (data / "annotations").is_dir():
        return False
    return all(
        any((data / name).is_dir() for name in names) for names in _CORPUS_DIR_NAMES
    )


def discover_main_checkout_data(project_root: Path) -> Path | None:
    """Return sibling main checkout ``data/`` when this is an experiment worktree.

    Layout expected when experiment worktrees sit next to the main checkout::

        attribute-based-person-retrieval-core/data          # shared
        attribute-based-person-retrieval-core-e34/          # worktree
    """
    root = project_root.resolve()
    main = (root.parent / MAIN_CHECKOUT_DIRNAME).resolve()
    if main == root:
        return None
    candidate = main / "data"
    if dataset_looks_present(candidate):
        return candidate
    return None


def discover_kaggle_input_data() -> Path | None:
    """Return a Kaggle input tree that already holds organizer corpora.

    Script kernels may mount datasets as ``/kaggle/input/<slug>`` or nested
    ``/kaggle/input/datasets/<user>/<slug>``. Walk for ``annotations/`` and
    accept the parent (or its parent) when all three corpora are present.
    """
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return None
    seen: set[Path] = set()
    for ann_dir in sorted(input_root.rglob("annotations")):
        if not ann_dir.is_dir():
            continue
        for candidate in (ann_dir.parent, ann_dir.parent.parent):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if dataset_looks_present(candidate):
                return candidate
    return None


def resolve_data_root(configured: Path, project_root: Path) -> Path:
    """Resolve ``data.root`` for shared worktree datasets.

    Order: ``WACV_DATA_ROOT`` env → absolute YAML path → local relative path
    when the dataset is already there → Kaggle input corpora when present →
    sibling main checkout ``data/`` when present → otherwise the local
    relative path (download target).

    Prefer a junction/symlink named ``data`` in new worktrees (see AGENTS.md).
    """
    override = os.environ.get(DATA_ROOT_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if configured.is_absolute():
        return configured.resolve()
    resolved = (project_root / configured).resolve()
    if dataset_looks_present(resolved):
        return resolved
    kaggle = discover_kaggle_input_data()
    if kaggle is not None:
        return kaggle
    shared = discover_main_checkout_data(project_root)
    if shared is not None:
        return shared
    return resolved
