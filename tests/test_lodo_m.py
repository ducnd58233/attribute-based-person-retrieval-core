"""Unit tests for LODO-M domain filters and query rebuild."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.wacv.data import (
    build_lodo_m_split,
    domain_from_image,
    filter_gt_by_domains,
)


def _toy_gt() -> pd.DataFrame:
    rows = [
        ("PA100k/a.jpg", 1, 0, 0),
        ("PA100k/b.jpg", 0, 1, 0),
        ("PETA/c.jpg", 1, 0, 0),
        ("Market1501/d.jpg", 1, 1, 0),
        ("Market1501/e.jpg", 1, 1, 0),
        ("Market1501/f.jpg", 0, 0, 1),
    ]
    return pd.DataFrame(rows, columns=["image", "a", "b", "c"])


def test_domain_from_image() -> None:
    assert domain_from_image("Market1501/bounding_box_train/x.jpg") == "Market1501"
    with pytest.raises(ValueError, match="Unknown domain"):
        domain_from_image("Other/x.jpg")


def test_filter_gt_drops_market_when_pa100k_peta() -> None:
    filtered = filter_gt_by_domains(_toy_gt(), ["PA100k", "PETA"])
    domains = [domain_from_image(p) for p in filtered["image"]]
    assert domains == ["PA100k", "PA100k", "PETA"]
    assert "Market1501" not in domains


def test_filter_gt_none_keeps_all() -> None:
    gt = _toy_gt()
    assert len(filter_gt_by_domains(gt, None)) == len(gt)


def test_build_lodo_m_queries_from_market_subset(tmp_path: Path) -> None:
    del tmp_path  # unused; keeps pytest fixture signature available
    gallery, queries = build_lodo_m_split(_toy_gt(), ("a", "b", "c"))
    assert len(gallery) == 3
    assert all(domain_from_image(p) == "Market1501" for p in gallery["image"])
    # Unique vectors: (1,1,0) and (0,0,1) in first-occurrence order.
    expected = np.asarray([[1, 1, 0], [0, 0, 1]], dtype=np.int64)
    np.testing.assert_array_equal(queries, expected)
