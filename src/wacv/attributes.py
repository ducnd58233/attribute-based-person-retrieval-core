"""Attribute column names derived from UPAR gt.csv headers.

The challenge's attribute order is the order of columns in each gt.csv after
``image``. Do not hardcode the list — always read it from the dataset file.
"""

from __future__ import annotations

from pathlib import Path


def attribute_names_from_gt(csv_path: Path) -> tuple[str, ...]:
    """Return attribute names from a gt.csv header (everything except ``image``)."""
    header = Path(csv_path).read_text(encoding="utf-8").splitlines()[0]
    columns = [c.strip() for c in header.lstrip("# ").split(",")]
    names = tuple(name for name in columns if name and name != "image")
    if not names:
        raise ValueError(f"no attribute columns found in {csv_path}")
    return names


def attribute_names_from_columns(
    columns: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Drop the ``image`` column from an already-parsed gt header/frame columns."""
    names = tuple(name for name in columns if name and name != "image")
    if not names:
        raise ValueError("no attribute columns found")
    return names
