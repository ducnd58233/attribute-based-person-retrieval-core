from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / ".cache" / "UPAR-Challenge-2027"
STARTER_URL = "https://github.com/speckean/UPAR-Challenge-2027.git"


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


def copy_annotations(starter: Path) -> None:
    src = starter / "data" / "annotations"
    dst = DATA / "annotations"
    if dst.is_dir():
        print(f"annotations already present: {dst}")
        return
    DATA.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    print(f"copied annotations -> {dst}")


def remap_peta(starter: Path) -> None:
    peta = DATA / "PETA"
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
        rel = file.relative_to(DATA).as_posix()
        dest = DATA / mapping[rel]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(file, dest)
    print(f"PETA remap done -> {images}")


def download_images(starter: Path) -> None:
    cmd = [
        sys.executable,
        str(starter / "download_datasets.py"),
        "--data-dir",
        str(DATA),
    ]
    print("downloading Market1501 / PA100K / PETA...")
    # BTC fails on Windows at PETA remap; zip/extract may already be done.
    result = subprocess.run(cmd, cwd=str(starter), check=False)
    if result.returncode != 0:
        print("BTC script exited non-zero (expected on Windows PETA remap)")
    remap_peta(starter)


def main() -> None:
    starter = clone_starter()
    copy_annotations(starter)
    download_images(starter)
    print(f"Done under {DATA}")


if __name__ == "__main__":
    main()
