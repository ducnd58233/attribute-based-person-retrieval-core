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
    parser.add_argument(
        "--tta",
        nargs="+",
        default=None,
        metavar="VIEW",
        help=(
            "Override inference TTA views written into model_card.json "
            "(e.g. identity, or identity hflip). Default: keep config values. "
            "Use a shorter list when packaging for Codabench's 1-hour wall clock."
        ),
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
    weight_format = "state_dict"
    image_h = int(config.data.image_size[0])
    image_w = int(config.data.image_size[1])
    if config.model.name.startswith("clip_") or config.model.name.startswith("dinov2_"):
        # TorchScript avoids vendoring open_clip / DINOv2 hub code into the eval image.
        model.eval()
        example = torch.zeros(1, 3, image_h, image_w)
        with torch.inference_mode():
            traced = torch.jit.trace(model, example)
        traced.save(str(destination))
        weight_format = "torchscript"
    else:
        torch.save(model.state_dict(), destination)
    return {
        "name": config.model.name,
        "dropout": config.model.dropout,
        "num_attributes": num_attributes,
        "hard_query_adapter": bool(config.model.hard_query_adapter),
        "adapter_dim": int(config.model.adapter_dim),
        "source_checkpoint": str(checkpoint),
        "weight_format": weight_format,
        "image_size": [image_h, image_w],
        "image_mean": (
            [0.48145466, 0.4578275, 0.40821073]
            if config.model.name.startswith("clip_")
            else [0.485, 0.456, 0.406]
        ),
        "image_std": (
            [0.26862954, 0.26130258, 0.27577711]
            if config.model.name.startswith("clip_")
            else [0.229, 0.224, 0.225]
        ),
        "ranking_distance": config.ranking.distance,
        "ranking_temperature": float(config.ranking.temperature),
        "ranking_alpha": float(config.ranking.alpha),
        "ranking_group_mode": config.ranking.group_mode,
        "ranking_calibration": config.ranking.calibration,
        "ranking_query_norm": config.ranking.query_norm,
        "ranking_query_norm_tau": float(config.ranking.query_norm_tau),
        "clear_fusion": float(config.ranking.clear_fusion),
        "inference_tta": list(config.inference.tta),
        "inference_tta_scale": float(config.inference.tta_scale),
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
    from src.wacv.utils import configure_default_logging

    configure_default_logging()
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
    from src.wacv.config import load_config as _load_config

    packed_config = _load_config(args.config)
    calib_candidate = (
        packed_config.run.output_dir / packed_config.run.name / "calibration.json"
    )
    if calib_candidate.is_file():
        shutil.copy2(calib_candidate, assets / "calibration.json")
        card["ranking_calibration"] = "per_attribute"
    if args.tta is not None:
        views = [str(view) for view in args.tta]
        if not views:
            raise ValueError("--tta requires at least one view name")
        card["inference_tta"] = views
    (assets / "model_card.json").write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    zip_path = out.parent / f"{out.name}.zip"
    _zip_submission(out, zip_path)
    print(json.dumps({"out": str(out), "zip": str(zip_path), "card": card}, indent=2))


if __name__ == "__main__":
    main()
