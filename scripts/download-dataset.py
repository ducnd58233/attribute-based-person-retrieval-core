from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "UPAR-Challenge-2027"
STARTER_URL = "https://github.com/speckean/UPAR-Challenge-2027.git"
DATA_ROOT_ENV = "WACV_DATA_ROOT"


def resolve_data_dir() -> Path:
    """Prefer ``WACV_DATA_ROOT`` so worktrees share the main checkout dataset."""
    override = os.environ.get(DATA_ROOT_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (ROOT / "data").resolve()


def _dataset_looks_present(data: Path) -> bool:
    """True when organizer annotations + at least one image corpus exist."""
    annotations = data / "annotations"
    image_roots = (
        data / "Market1501",
        data / "Market-1501-v15.09.15",
        data / "PA100k",
        data / "PETA",
    )
    return annotations.is_dir() and any(path.is_dir() for path in image_roots)


def _sibling_main_data() -> Path | None:
    """Main submodule ``data/`` when this checkout is an experiment worktree."""
    main = ROOT.parent / "attribute-based-person-retrieval-core"
    if not main.is_dir() or main.resolve() == ROOT.resolve():
        return None
    candidate = main / "data"
    if _dataset_looks_present(candidate):
        return candidate.resolve()
    return None


def clone_starter() -> Path:
    if (CACHE / "download_datasets.py").exists():
        return CACHE
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        shutil.rmtree(CACHE)
    subprocess.run(
        ["git", "clone", "--depth", "1", STARTER_URL, str(CACHE)],
        check=True,
    )
    return CACHE


def copy_annotations(starter: Path, data: Path) -> None:
    src = starter / "data" / "annotations"
    dst = data / "annotations"
    if dst.is_dir():
        print(f"annotations already present: {dst}")
        return
    data.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    print(f"copied annotations -> {dst}")


def remap_peta(starter: Path, data: Path) -> None:
    import numpy as np
    from tqdm import tqdm

    peta = data / "PETA"
    images = peta / "images"
    if not peta.is_dir():
        print("PETA folder missing; skip remap")
        return
    images.mkdir(parents=True, exist_ok=True)

    mapping_file = starter / "peta_file_mapping.txt"
    mapping = {
        row[0]: row[1] for row in np.genfromtxt(mapping_file, dtype=str, delimiter=",")
    }

    files = [f for f in peta.glob("*/*/*/*") if f.suffix.lower() != ".txt"]
    if not files:
        print(f"PETA already remapped (or empty): {images}")
        return

    for file in tqdm(files, desc="Remap PETA"):
        rel = file.relative_to(data).as_posix()
        dest = data / mapping[rel]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(file, dest)
    print(f"PETA remap done -> {images}")


def download_images(starter: Path, data: Path) -> None:
    cmd = [
        sys.executable,
        str(starter / "download_datasets.py"),
        "--data-dir",
        str(data),
    ]
    print("downloading Market1501 / PA100K / PETA...")
    # BTC fails on Windows at PETA remap; zip/extract may already be done.
    result = subprocess.run(cmd, cwd=str(starter), check=False)
    if result.returncode != 0:
        print("BTC script exited non-zero (expected on Windows PETA remap)")
    remap_peta(starter, data)


def main() -> None:
    data = resolve_data_dir()
    if os.environ.get(DATA_ROOT_ENV, "").strip():
        print(f"{DATA_ROOT_ENV}={data}")
    if _dataset_looks_present(data):
        print(f"dataset already present under {data}; skipping download")
        print(
            f"(unset {DATA_ROOT_ENV} and remove data/ only if you "
            "truly need a re-fetch)"
        )
        return
    shared = _sibling_main_data()
    if shared is not None and data.resolve() != shared:
        print(
            "refusing to download a second organizer dataset copy.\n"
            f"  this worktree would write: {data}\n"
            f"  main checkout already has: {shared}\n"
            "  fix: ln -sfn <main>/data data   OR   export "
            f"{DATA_ROOT_ENV}={shared}\n"
            "  see AGENTS.md 'Experiment worktrees: shared data root'.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    starter = clone_starter()
    copy_annotations(starter, data)
    download_images(starter, data)
    print(f"Done under {data}")


if __name__ == "__main__":
    main()
