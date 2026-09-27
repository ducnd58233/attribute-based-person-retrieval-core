"""Attribute column names derived from UPAR gt.csv headers.

The challenge's attribute order is the order of columns in each gt.csv after
``image``. Do not hardcode the list — always read it from the dataset file.

Exclusive-group index maps for categorical scoring follow the official 40-name
order in PROBLEM-DETAILS.md (Age, Hair, colours, lower type).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExclusiveAttributeGroup:
    """One mutually exclusive attribute group used by categorical scoring."""

    name: str
    indices: tuple[int, ...]
    has_none: bool = False


# Official UPAR order (PROBLEM-DETAILS.md § Attributes). Indices only — names
# are validated at runtime against gt.csv so column reorders fail loudly.
EXCLUSIVE_ATTRIBUTE_GROUPS: tuple[ExclusiveAttributeGroup, ...] = (
    ExclusiveAttributeGroup("Age", (0, 1, 2)),
    ExclusiveAttributeGroup("Hair", (4, 5, 6), has_none=True),
    ExclusiveAttributeGroup("UpperColor", tuple(range(8, 20))),
    ExclusiveAttributeGroup("LowerColor", tuple(range(21, 33))),
    ExclusiveAttributeGroup("LowerType", (33, 34)),
)

_EXPECTED_GROUP_NAMES: dict[str, tuple[str, ...]] = {
    "Age": ("Age-Young", "Age-Adult", "Age-Old"),
    "Hair": ("Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald"),
    "UpperColor": (
        "UpperBody-Color-Black",
        "UpperBody-Color-Blue",
        "UpperBody-Color-Brown",
        "UpperBody-Color-Green",
        "UpperBody-Color-Grey",
        "UpperBody-Color-Orange",
        "UpperBody-Color-Pink",
        "UpperBody-Color-Purple",
        "UpperBody-Color-Red",
        "UpperBody-Color-White",
        "UpperBody-Color-Yellow",
        "UpperBody-Color-Other",
    ),
    "LowerColor": (
        "LowerBody-Color-Black",
        "LowerBody-Color-Blue",
        "LowerBody-Color-Brown",
        "LowerBody-Color-Green",
        "LowerBody-Color-Grey",
        "LowerBody-Color-Orange",
        "LowerBody-Color-Pink",
        "LowerBody-Color-Purple",
        "LowerBody-Color-Red",
        "LowerBody-Color-White",
        "LowerBody-Color-Yellow",
        "LowerBody-Color-Other",
    ),
    "LowerType": ("LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress"),
}


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


def exclusive_attribute_groups(
    attribute_names: tuple[str, ...] | None = None,
) -> tuple[ExclusiveAttributeGroup, ...]:
    """Return exclusive groups; optionally validate against ``attribute_names``."""
    if attribute_names is not None:
        if len(attribute_names) < 40:
            raise ValueError(
                f"expected at least 40 attributes, got {len(attribute_names)}"
            )
        for group in EXCLUSIVE_ATTRIBUTE_GROUPS:
            expected = _EXPECTED_GROUP_NAMES[group.name]
            actual = tuple(attribute_names[i] for i in group.indices)
            if actual != expected:
                raise ValueError(
                    f"group {group.name!r} name mismatch: {actual} != {expected}"
                )
    return EXCLUSIVE_ATTRIBUTE_GROUPS


def independent_attribute_indices(num_attributes: int = 40) -> tuple[int, ...]:
    """Indices scored with Bernoulli NLL under group-categorical mode."""
    covered: set[int] = set()
    for group in EXCLUSIVE_ATTRIBUTE_GROUPS:
        covered.update(group.indices)
    return tuple(i for i in range(num_attributes) if i not in covered)
