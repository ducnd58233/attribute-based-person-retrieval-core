"""Assemble build/submission/ from the tracked template + a training checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from src.wacv.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ZIP_EXCLUDED_DIR_NAMES = {"__pycache__", ".git"}
_ZIP_EXCLUDED_FILE_NAMES = {".DS_Store"}


class _LogitsAndFeatures(torch.nn.Module):
    """Traceable wrapper returning ``(logits, pooled features)`` for smoothing."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        from src.wacv.evaluation import FeatureEncoder

        if not isinstance(model, FeatureEncoder):
            raise TypeError(
                f"{type(model).__name__} has no encode_features; "
                "neighbour_smooth supports the CLIP and DINOv2 classifiers"
            )
        self.model = model
        self.head = model.get_submodule("head")

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.encode_features(images)
        return self.head(features), features


def parse_args() -> argparse.Namespace:
    from src.wacv.config import DEFAULT_CONFIG_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trainer checkpoint (full payload or state_dict)",
    )
    parser.add_argument(
        "--member",
        type=Path,
        action="append",
        default=[],
        metavar="CHECKPOINT",
        help=(
            "Extra checkpoint built from the same --config; the package then "
            "traces an equal-weight probability average of all checkpoints"
        ),
    )
    parser.add_argument(
        "--from-member-dir",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "Prebuilt KEEP/submission dir with model.pt + model_card.json "
            "(or assets/). Heterogeneous image_size/mean/std OK; equal-weight "
            "average via run.py per-member transforms. Requires "
            "--no-neighbour-smooth. Do not combine with --checkpoint/--member."
        ),
    )
    parser.add_argument(
        "--member-weight",
        type=float,
        action="append",
        default=[],
        metavar="W",
        help=(
            "Averaging weight of each --from-member-dir, in the same order "
            "(repeat once per directory; normalised to sum 1). Default: equal."
        ),
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=0,
        help="inference_decode_workers in model_card.json (0 = serial decode)",
    )
    parser.add_argument(
        "--weights-dtype",
        choices=("fp32", "fp16"),
        default="fp32",
        help=(
            "Store TorchScript weights in fp16 (half the zip and load time); "
            "run.py upcasts them to fp32 after loading"
        ),
    )
    parser.add_argument(
        "--amp",
        choices=("none", "bf16", "fp16"),
        default="none",
        help="inference_amp in model_card.json (CUDA autocast at inference)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
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
    parser.add_argument(
        "--no-neighbour-smooth",
        action="store_true",
        help=(
            "Force latent kNN smoothing off, even when the config sets "
            "ranking.neighbour_smooth. Smoothing is off unless the config asks."
        ),
    )
    return parser.parse_args()


def _resolve_neighbour_smooth(
    config: ExperimentConfig, *, disable: bool
) -> dict[str, float | int] | None:
    """Return the config's smoothing block, or None when it is unset or disabled.

    Smoothing is a kNN pass over the gallery being ranked, which on Codabench is
    the hidden test gallery; the organizers do not allow it there, so a package
    only carries it when the config sets ``ranking.neighbour_smooth`` itself.
    """
    if disable:
        return None
    supported = config.model.name.startswith("clip_") or config.model.name.startswith(
        "dinov2_"
    )
    if not supported:
        if config.ranking.neighbour_smooth is not None:
            raise ValueError(
                "ranking.neighbour_smooth needs a CLIP or DINOv2 checkpoint "
                f"(got {config.model.name})"
            )
        return None
    smooth = config.ranking.neighbour_smooth
    return None if smooth is None else smooth.model_dump()


def _export_weights(
    checkpoints: list[Path],
    config_path: Path,
    destination: Path,
    *,
    disable_neighbour_smooth: bool,
) -> dict[str, Any]:
    from src.wacv.attributes import attribute_names_from_gt
    from src.wacv.checkpoint import load_model_weights
    from src.wacv.config import load_config
    from src.wacv.models import build_model
    from src.wacv.models.ensemble import ProbabilityAverage
    from src.wacv.normalization import normalization_for_model

    config = load_config(config_path)
    gt_path = config.data.root / "annotations" / "task1" / "train" / "gt.csv"
    num_attributes = len(attribute_names_from_gt(gt_path))
    members: list[torch.nn.Module] = []
    for checkpoint in checkpoints:
        member = build_model(
            config.model.model_copy(update={"pretrained": False}),
            num_attributes=num_attributes,
        )
        load_model_weights(
            checkpoint.resolve(), member, map_location=torch.device("cpu")
        )
        members.append(member)
    model: torch.nn.Module = members[0]
    if len(members) > 1:
        if not disable_neighbour_smooth:
            raise ValueError(
                "a probability-average package needs --no-neighbour-smooth"
            )
        model = ProbabilityAverage(members, [1.0 / len(members)] * len(members))
    destination.parent.mkdir(parents=True, exist_ok=True)
    weight_format = "state_dict"
    image_h, image_w = config.data.image_size
    stats = normalization_for_model(config.model.name)
    neighbour_smooth = _resolve_neighbour_smooth(
        config, disable=disable_neighbour_smooth
    )
    if config.model.name.startswith("clip_") or config.model.name.startswith("dinov2_"):
        # TorchScript avoids vendoring open_clip / DINOv2 hub code into the eval image.
        model.eval()
        example = torch.zeros(1, 3, image_h, image_w)
        traced_module = (
            _LogitsAndFeatures(model) if neighbour_smooth is not None else model
        )
        with torch.inference_mode():
            traced = cast(
                torch.jit.ScriptModule, torch.jit.trace(traced_module, example)
            )
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
        "source_checkpoint": str(checkpoints[0]),
        "source_checkpoints": [str(path) for path in checkpoints],
        "weight_format": weight_format,
        "image_size": [image_h, image_w],
        "image_mean": list(stats.mean),
        "image_std": list(stats.std),
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
        "neighbour_smooth": neighbour_smooth,
    }


def _resolve_member_dir_assets(member_dir: Path) -> tuple[Path, Path]:
    """Return (model.pt, model_card.json) under a KEEP or packaged submission dir."""
    root = member_dir.resolve()
    candidates = (
        (root / "model.pt", root / "model_card.json"),
        (root / "assets" / "model.pt", root / "assets" / "model_card.json"),
    )
    for weights, card_path in candidates:
        if weights.is_file() and card_path.is_file():
            return weights, card_path
    raise FileNotFoundError(
        f"{root} needs model.pt + model_card.json (root or assets/)"
    )


def _copy_or_link(source: Path, destination: Path) -> None:
    """Prefer a hardlink to save disk; fall back to a full copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _resolve_member_weights(count: int, requested: list[float]) -> list[float]:
    """Normalised averaging weights: equal when none are given."""
    if not requested:
        return [1.0 / count] * count
    if len(requested) != count:
        raise ValueError(
            f"--member-weight needs one value per --from-member-dir ({count}), "
            f"got {len(requested)}"
        )
    if any(weight <= 0.0 for weight in requested):
        raise ValueError("--member-weight values must be positive")
    total = sum(requested)
    return [weight / total for weight in requested]


def _assemble_from_member_dirs(
    member_dirs: list[Path],
    assets_dir: Path,
    *,
    tta: list[str] | None,
    decode_workers: int,
    amp: str,
    member_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Build a multi-TorchScript package with per-member preprocess fields."""
    if len(member_dirs) < 2:
        raise ValueError("--from-member-dir needs at least two directories")
    members: list[dict[str, Any]] = []
    ranking_source: dict[str, Any] | None = None
    weights = _resolve_member_weights(len(member_dirs), member_weights or [])
    for index, member_dir in enumerate(member_dirs):
        weights_path, card_path = _resolve_member_dir_assets(member_dir)
        source_card = json.loads(card_path.read_text(encoding="utf-8"))
        if ranking_source is None:
            ranking_source = source_card
        if str(source_card.get("weight_format", "state_dict")) != "torchscript":
            raise ValueError(
                f"{member_dir}: only TorchScript members are supported "
                f"(got weight_format={source_card.get('weight_format')!r})"
            )
        if source_card.get("neighbour_smooth") is not None:
            raise ValueError(
                f"{member_dir}: source card has neighbour_smooth; "
                "refuse to package gallery smoothing"
            )
        dest_name = f"model_{index}.pt"
        _copy_or_link(weights_path, assets_dir / dest_name)
        member_tta = (
            list(tta)
            if tta is not None
            else [str(view) for view in source_card.get("inference_tta", ["identity"])]
        )
        if not member_tta:
            raise ValueError(f"{member_dir}: empty inference_tta")
        members.append(
            {
                "weights": dest_name,
                "name": str(source_card["name"]),
                "num_attributes": int(source_card["num_attributes"]),
                "image_size": list(source_card["image_size"]),
                "image_mean": list(source_card["image_mean"]),
                "image_std": list(source_card["image_std"]),
                "weight": weights[index],
                "weight_format": "torchscript",
                "weight_dtype": str(source_card.get("weight_dtype", "fp32")),
                "inference_tta": member_tta,
                "source_dir": str(member_dir.resolve()),
            }
        )
    assert ranking_source is not None
    first = members[0]
    return {
        "name": "torchscript_members",
        "num_attributes": int(first["num_attributes"]),
        "weight_format": "torchscript_members",
        "source_checkpoints": [str(path.resolve()) for path in member_dirs],
        "members": members,
        "image_size": list(first["image_size"]),
        "image_mean": list(first["image_mean"]),
        "image_std": list(first["image_std"]),
        "ranking_distance": ranking_source.get("ranking_distance", "hybrid"),
        "ranking_temperature": float(ranking_source.get("ranking_temperature", 1.0)),
        "ranking_alpha": float(ranking_source.get("ranking_alpha", 0.5)),
        "ranking_group_mode": ranking_source.get("ranking_group_mode", "independent"),
        "ranking_calibration": ranking_source.get("ranking_calibration", "none"),
        "ranking_query_norm": ranking_source.get("ranking_query_norm", "dual_softmax"),
        "ranking_query_norm_tau": float(
            ranking_source.get("ranking_query_norm_tau", 0.05)
        ),
        "clear_fusion": float(ranking_source.get("clear_fusion", 0.5)),
        "inference_tta": (
            list(tta) if tta is not None else list(members[0]["inference_tta"])
        ),
        "inference_tta_scale": float(ranking_source.get("inference_tta_scale", 1.15)),
        "neighbour_smooth": None,
        "inference_decode_workers": int(decode_workers),
        "inference_amp": str(amp),
        "weight_dtype": "fp32",
    }


def _convert_torchscript_to_fp16(path: Path, card: dict[str, Any]) -> None:
    """Rewrite a TorchScript package with fp16 parameters and buffers.

    CLIP's released weights are fp16 already, so the frozen tower loses
    nothing; only trained adapters and heads are rounded.
    """
    if card["weight_format"] != "torchscript":
        raise ValueError(
            "--weights-dtype fp16 needs a TorchScript (CLIP/DINOv2) package"
        )
    module = torch.jit.load(str(path), map_location="cpu")
    module.half()
    module.save(str(path))


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


def _check_package_inputs(args: argparse.Namespace, *, use_member_dirs: bool) -> None:
    """Refuse argument mixes the two packaging paths cannot honour."""
    if use_member_dirs:
        if args.checkpoint is not None or args.member:
            raise ValueError(
                "--from-member-dir cannot be combined with --checkpoint/--member"
            )
        if not args.no_neighbour_smooth:
            raise ValueError(
                "a multi-preprocess member package needs --no-neighbour-smooth"
            )
    elif args.checkpoint is None:
        raise ValueError("provide --checkpoint or at least two --from-member-dir")
    if args.member_weight and not use_member_dirs:
        raise ValueError("--member-weight only applies to --from-member-dir")


def _export_checkpoint_package(
    args: argparse.Namespace, assets: Path
) -> dict[str, Any]:
    """Export one or several trainer checkpoints into a single TorchScript file."""
    assert args.checkpoint is not None
    card = _export_weights(
        [args.checkpoint, *args.member],
        args.config,
        assets / "model.pt",
        disable_neighbour_smooth=bool(args.no_neighbour_smooth),
    )
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
    card["weight_dtype"] = str(args.weights_dtype)
    if args.weights_dtype == "fp16":
        _convert_torchscript_to_fp16(assets / "model.pt", card)
    card["inference_decode_workers"] = int(args.decode_workers)
    card["inference_amp"] = str(args.amp)
    return card


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

    member_dirs = [path.resolve() for path in args.from_member_dir]
    use_member_dirs = bool(member_dirs)
    _check_package_inputs(args, use_member_dirs=use_member_dirs)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(run_py, out / "run.py")
    shutil.copy2(metadata, out / "metadata.yaml")

    assets = out / "assets"
    assets.mkdir()
    if use_member_dirs:
        card = _assemble_from_member_dirs(
            member_dirs,
            assets,
            tta=[str(view) for view in args.tta] if args.tta is not None else None,
            decode_workers=int(args.decode_workers),
            amp=str(args.amp),
            member_weights=[float(weight) for weight in args.member_weight],
        )
    else:
        card = _export_checkpoint_package(args, assets)
    if card["neighbour_smooth"] is not None and "identity" not in card["inference_tta"]:
        raise ValueError("neighbour_smooth takes its features from the identity view")
    (assets / "model_card.json").write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    zip_path = out.parent / f"{out.name}.zip"
    _zip_submission(out, zip_path)
    print(json.dumps({"out": str(out), "zip": str(zip_path), "card": card}, indent=2))


if __name__ == "__main__":
    main()
