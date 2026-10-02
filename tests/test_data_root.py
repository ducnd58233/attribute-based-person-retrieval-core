"""Shared data-root resolution for experiment git worktrees."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.wacv.config import (
    DATA_ROOT_ENV,
    dataset_looks_present,
    discover_main_checkout_data,
    resolve_data_root,
)


def _plant_dataset(data: Path) -> None:
    (data / "annotations").mkdir(parents=True)
    (data / "Market1501").mkdir(parents=True)
    (data / "annotations" / "dummy.txt").write_text("x", encoding="utf-8")


def test_dataset_looks_present_requires_annotations_and_corpus(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    assert not dataset_looks_present(data)
    (data / "annotations").mkdir()
    assert not dataset_looks_present(data)
    (data / "PA100k").mkdir()
    assert dataset_looks_present(data)


def test_resolve_prefers_env_over_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "worktree"
    local.mkdir()
    shared = tmp_path / "shared-data"
    _plant_dataset(shared)
    monkeypatch.setenv(DATA_ROOT_ENV, str(shared))
    assert resolve_data_root(Path("data"), local) == shared.resolve()


def test_resolve_uses_sibling_main_when_local_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    workspace = tmp_path / "workspace"
    main = workspace / "attribute-based-person-retrieval-core"
    worktree = workspace / "attribute-based-person-retrieval-core-e99"
    main.mkdir(parents=True)
    worktree.mkdir(parents=True)
    _plant_dataset(main / "data")

    assert discover_main_checkout_data(worktree) == (main / "data").resolve()
    assert resolve_data_root(Path("data"), worktree) == (main / "data").resolve()


def test_resolve_keeps_local_when_dataset_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    workspace = tmp_path / "workspace"
    main = workspace / "attribute-based-person-retrieval-core"
    worktree = workspace / "attribute-based-person-retrieval-core-e99"
    main.mkdir(parents=True)
    worktree.mkdir(parents=True)
    _plant_dataset(main / "data")
    _plant_dataset(worktree / "data")

    assert resolve_data_root(Path("data"), worktree) == (worktree / "data").resolve()


def test_main_checkout_does_not_discover_self(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    workspace = tmp_path / "workspace"
    main = workspace / "attribute-based-person-retrieval-core"
    main.mkdir(parents=True)
    _plant_dataset(main / "data")
    assert discover_main_checkout_data(main) is None
    assert resolve_data_root(Path("data"), main) == (main / "data").resolve()


def test_absolute_configured_path_skips_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    workspace = tmp_path / "workspace"
    main = workspace / "attribute-based-person-retrieval-core"
    worktree = workspace / "attribute-based-person-retrieval-core-e99"
    other = tmp_path / "other-data"
    main.mkdir(parents=True)
    worktree.mkdir(parents=True)
    _plant_dataset(main / "data")
    other.mkdir()
    assert resolve_data_root(other, worktree) == other.resolve()
