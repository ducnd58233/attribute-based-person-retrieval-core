from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = ROOT / ".cache" / "UPAR-Challenge-2027"
STARTER_URL = "https://github.com/speckean/UPAR-Challenge-2027.git"


def resolve_data_dir() -> Path:
    """Same rules as ``load_config``: env → local → sibling main checkout."""
    from src.wacv.data_root import resolve_data_root

    return resolve_data_root(Path("data"), ROOT)


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
    from src.wacv.data_root import (
        DATA_ROOT_ENV,
        MAIN_CHECKOUT_DIRNAME,
        dataset_looks_present,
    )

    data = resolve_data_dir()
    local = (ROOT / "data").resolve()
    if os.environ.get(DATA_ROOT_ENV, "").strip():
        print(f"{DATA_ROOT_ENV}={data}")
    elif data != local:
        print(f"using shared data root (sibling main checkout): {data}")
    if dataset_looks_present(data):
        print(f"dataset already present under {data}; skipping download")
        print(
            f"(unset {DATA_ROOT_ENV} and remove data/ only if you "
            "truly need a re-fetch)"
        )
        return
    # Experiment worktrees must never materialize a second organizer copy.
    main_checkout = (ROOT.parent / MAIN_CHECKOUT_DIRNAME).resolve()
    if main_checkout.is_dir() and main_checkout != ROOT.resolve():
        data = main_checkout / "data"
        print(
            f"worktree download redirected to main checkout data/: {data}\n"
            "  (see AGENTS.md 'Experiment worktrees: shared data root')"
        )
    starter = clone_starter()
    copy_annotations(starter, data)
    download_images(starter, data)
    print(f"Done under {data}")


if __name__ == "__main__":
    main()
