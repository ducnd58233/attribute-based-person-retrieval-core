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

    Layout expected under the parent workspace::

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


def resolve_data_root(configured: Path, project_root: Path) -> Path:
    """Resolve ``data.root`` for shared worktree datasets.

    Order: ``WACV_DATA_ROOT`` env → absolute YAML path → local relative path
    when the dataset is already there → sibling main checkout ``data/`` when
    present → otherwise the local relative path (download target).

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
    shared = discover_main_checkout_data(project_root)
    if shared is not None:
        return shared
    return resolved
