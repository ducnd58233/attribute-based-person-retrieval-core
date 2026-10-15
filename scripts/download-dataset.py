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
_KAGGLE_ROOT = Path("/kaggle")


def resolve_data_dir() -> Path:
    """Same rules as ``load_config``: env → local → sibling main checkout."""
    from src.wacv.data_root import resolve_data_root

    return resolve_data_root(Path("data"), ROOT)


def _on_kaggle() -> bool:
    return _KAGGLE_ROOT.is_dir()


def find_kaggle_input_data() -> Path | None:
    """Return a /kaggle/input tree that already holds organizer corpora."""
    from src.wacv.data_root import discover_kaggle_input_data

    return discover_kaggle_input_data()


def find_vendored_starter() -> Path | None:
    """Locate a shipped UPAR starter (staging, cache, or Kaggle input)."""
    candidates: list[Path] = [CACHE, ROOT / ".cache" / "UPAR-Challenge-2027"]
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(sorted(input_root.glob("*/UPAR-Challenge-2027")))
        candidates.extend(sorted(input_root.glob("*/.cache/UPAR-Challenge-2027")))
        candidates.extend(sorted(input_root.glob("*/starter")))
        # Any attached dataset that embeds the starter script.
        for script in sorted(input_root.rglob("download_datasets.py")):
            candidates.append(script.parent)
    for path in candidates:
        if (path / "download_datasets.py").is_file():
            return path
    return None


def clone_starter() -> Path:
    """Resolve the UPAR starter kit without outbound git when vendored."""
    vendored = find_vendored_starter()
    if vendored is not None:
        print(f"using vendored UPAR starter: {vendored}")
        return vendored
    if _on_kaggle():
        raise SystemExit(
            "UPAR starter not vendored on Kaggle and git clone is disabled "
            "(kernel DNS cannot reach github.com). Attach "
            "wacv-upar-organizer-data (full corpora) or ship "
            ".cache/UPAR-Challenge-2027 in the staging folder / wheels dataset."
        )
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


def link_or_copy_data(src: Path, dst: Path) -> None:
    """Point ``dst`` at ``src`` (symlink preferred; copy as last resort).

    On Kaggle, never ``copytree`` the organizer corpora (multi-GB; OOMs the
    kernel). ``resolve_data_root`` can still read the input mount directly.
    """
    if dst.resolve() == src.resolve():
        return
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(src, target_is_directory=True)
        print(f"symlinked {dst} -> {src}")
    except OSError as exc:
        if _on_kaggle():
            print(
                f"symlink failed ({exc}); skipping copytree on Kaggle — "
                f"train/eval will use input mount via resolve_data_root ({src})"
            )
            return
        print(f"symlink failed; copying {src} -> {dst} (slow)")
        shutil.copytree(src, dst)


def copy_annotations(starter: Path, data: Path) -> None:
    src = starter / "data" / "annotations"
    # Offline pack: corpora live at <dataset>/data/annotations; starter kit
    # may ship without a nested data/ tree.
    if not src.is_dir():
        sibling = starter.parent / "data" / "annotations"
        if sibling.is_dir():
            src = sibling
    dst = data / "annotations"
    if dst.is_dir():
        print(f"annotations already present: {dst}")
        return
    if not src.is_dir():
        raise SystemExit(
            f"annotations missing under starter ({starter / 'data' / 'annotations'}) "
            f"and sibling pack ({starter.parent / 'data' / 'annotations'})"
        )
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
    if _on_kaggle():
        raise SystemExit(
            "Refusing network image download on Kaggle (DNS-dead kernels). "
            "Attach dataset wacv-upar-organizer-data with data/Market1501|"
            "PA100k|PETA|annotations under /kaggle/input/."
        )
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

    kaggle_data = find_kaggle_input_data()
    if kaggle_data is not None:
        local = (ROOT / "data").resolve()
        if not dataset_looks_present(local):
            local.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy_data(kaggle_data, local)
        print(f"dataset present via Kaggle input: {kaggle_data}")
        return

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
        if dataset_looks_present(data):
            print(f"dataset already present under {data}; skipping download")
            return
    starter = clone_starter()
    copy_annotations(starter, data)
    download_images(starter, data)
    print(f"Done under {data}")


if __name__ == "__main__":
    main()
