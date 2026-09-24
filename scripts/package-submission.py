"""Assemble build/submission/ from the tracked template + a training checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ZIP_EXCLUDED_DIR_NAMES = {"__pycache__", ".git"}
_ZIP_EXCLUDED_FILE_NAMES = {".DS_Store"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trainer checkpoint (full payload or state_dict)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "convnext_baseline.yml",
        help="Rebuild architecture for a clean weight export",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build" / "submission",
        help="Destination folder (contents are replaced)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "submission",
        help="Tracked run.py + metadata.yaml template",
    )
    return parser.parse_args()


def _export_weights(
    checkpoint: Path, config_path: Path, destination: Path
) -> dict[str, Any]:
    from src.wacv.attributes import attribute_names_from_gt
    from src.wacv.checkpoint import load_model_weights
    from src.wacv.config import load_config
    from src.wacv.models import build_model

    config = load_config(config_path)
    gt_path = config.data.root / "annotations" / "task1" / "train" / "gt.csv"
    num_attributes = len(attribute_names_from_gt(gt_path))
    model = build_model(
        config.model.model_copy(update={"pretrained": False}),
        num_attributes=num_attributes,
    )
    load_model_weights(checkpoint.resolve(), model, map_location=torch.device("cpu"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    return {
        "name": config.model.name,
        "dropout": config.model.dropout,
        "num_attributes": num_attributes,
        "source_checkpoint": str(checkpoint),
    }


def _zip_submission(source: Path, destination: Path) -> None:
    """Zip source's contents, not source itself, so run.py lands at the archive root."""
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if _ZIP_EXCLUDED_DIR_NAMES & set(relative.parts):
                continue
            if path.name in _ZIP_EXCLUDED_FILE_NAMES:
                continue
            archive.write(path, arcname=str(relative))


def main() -> None:
    args = parse_args()
    template = args.template.resolve()
    out = args.out.resolve()
    run_py = template / "run.py"
    metadata = template / "metadata.yaml"
    if not run_py.is_file() or not metadata.is_file():
        raise FileNotFoundError(
            f"Template missing run.py/metadata.yaml under {template}"
        )

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(run_py, out / "run.py")
    shutil.copy2(metadata, out / "metadata.yaml")

    assets = out / "assets"
    assets.mkdir()
    card = _export_weights(args.checkpoint, args.config, assets / "model.pt")
    (assets / "model_card.json").write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    zip_path = out.parent / f"{out.name}.zip"
    _zip_submission(out, zip_path)
    print(json.dumps({"out": str(out), "zip": str(zip_path), "card": card}, indent=2))


if __name__ == "__main__":
    main()
