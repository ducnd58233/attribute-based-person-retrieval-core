"""Dump fp16 probs/logits/embeddings (+ ids) for T28 artifact bundles.

Writes under ``runs/<run.name>/artifacts/``:
  - ``{split}_probs.npz``, ``{split}_logits.npz``, ``{split}_embeddings.npz``
  - ``{split}_ids.json`` (image path strings in row order)

Splits: ``id_val``, ``lodo_m``, and ``train_subset`` (fixed 20k, seed 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_SUBSET_SIZE = 20_000
TRAIN_SUBSET_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="defaults to runs/<run.name>/checkpoints/best.pt",
    )
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=TRAIN_SUBSET_SIZE,
    )
    return parser.parse_args()


@torch.inference_mode()
def _forward_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images = images.to(device, non_blocking=True)
    logits = model(images)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    probs = torch.sigmoid(logits)
    if hasattr(model, "encode_features"):
        embeddings = model.encode_features(images)  # type: ignore[operator]
    elif hasattr(model, "backbone"):
        features = model.backbone(images)  # type: ignore[operator]
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        embeddings = features
    else:
        embeddings = logits
    return (
        probs.float().cpu().numpy().astype(np.float16),
        logits.float().cpu().numpy().astype(np.float16),
        embeddings.float().cpu().numpy().astype(np.float16),
    )


def _dump_loader(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    split: str,
    image_ids: list[str],
) -> None:
    all_probs: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_embeds: list[np.ndarray] = []
    for images, _labels in loader:
        probs, logits, embeds = _forward_batch(model, images, device)
        all_probs.append(probs)
        all_logits.append(logits)
        all_embeds.append(embeds)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{split}_probs.npz", data=np.concatenate(all_probs))
    np.savez_compressed(
        out_dir / f"{split}_logits.npz", data=np.concatenate(all_logits)
    )
    np.savez_compressed(
        out_dir / f"{split}_embeddings.npz", data=np.concatenate(all_embeds)
    )
    (out_dir / f"{split}_ids.json").write_text(
        json.dumps(image_ids, indent=2) + "\n", encoding="utf-8"
    )


def _make_loader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


@dataclass(frozen=True)
class _DumpSplitCtx:
    """Shared dump context for one split."""

    model: torch.nn.Module
    data_root: Path
    transform: object
    attribute_names: tuple[str, ...]
    batch_size: int
    workers: int
    device: torch.device
    out_dir: Path


def _dump_gt_split(
    *,
    ctx: _DumpSplitCtx,
    ground_truth,
    split: str,
) -> None:
    from src.wacv.data import AttributeDataset

    dataset = AttributeDataset(
        ground_truth,
        ctx.data_root,
        ctx.transform,
        attribute_names=ctx.attribute_names,
    )
    loader = _make_loader(
        dataset,
        batch_size=ctx.batch_size,
        workers=ctx.workers,
        device=ctx.device,
    )
    image_ids = [str(path) for path in ground_truth["image"].tolist()]
    _dump_loader(
        model=ctx.model,
        loader=loader,
        device=ctx.device,
        out_dir=ctx.out_dir,
        split=split,
        image_ids=image_ids,
    )


def main() -> int:
    args = parse_args()
    from src.wacv.attributes import attribute_names_from_columns
    from src.wacv.checkpoint import load_model_weights
    from src.wacv.config import load_config
    from src.wacv.data import (
        AttributeDataset,
        build_lodo_m_split,
        filter_gt_by_domains,
        load_gt,
        transforms_for_model,
    )
    from src.wacv.engine import resolve_device, resolve_num_workers
    from src.wacv.models import build_model

    config = load_config(args.config)
    device = resolve_device(config.run.device)
    run_dir = config.run.output_dir / config.run.name
    checkpoint = (
        args.checkpoint
        if args.checkpoint is not None
        else run_dir / "checkpoints" / "best.pt"
    )
    val_root = config.data.root / "annotations" / "task2" / "val"
    ground_truth = load_gt(val_root / "gt.csv")
    attribute_names = attribute_names_from_columns(
        tuple(str(column) for column in ground_truth.columns)
    )
    inference_model = config.model.model_copy(update={"pretrained": False})
    model = build_model(inference_model, num_attributes=len(attribute_names)).to(device)
    load_model_weights(checkpoint.resolve(), model, map_location=device)
    model.eval()

    workers = resolve_num_workers(config.data.num_workers)
    image_size = (
        int(config.data.image_size[0]),
        int(config.data.image_size[1]),
    )
    transform = transforms_for_model(
        config.model.name, train=False, image_size=image_size
    )
    out_dir = run_dir / "artifacts"
    ctx = _DumpSplitCtx(
        model=model,
        data_root=config.data.root,
        transform=transform,
        attribute_names=attribute_names,
        batch_size=config.data.batch_size,
        workers=workers,
        device=device,
        out_dir=out_dir,
    )

    id_gt = filter_gt_by_domains(ground_truth, config.data.eval_domains)
    _dump_gt_split(ctx=ctx, ground_truth=id_gt, split="id_val")

    lodo_gt, _queries = build_lodo_m_split(ground_truth, attribute_names)
    _dump_gt_split(ctx=ctx, ground_truth=lodo_gt, split="lodo_m")

    train_root = config.data.root / "annotations" / "task2" / "train"
    train_gt = filter_gt_by_domains(
        load_gt(train_root / "gt.csv"), config.data.train_domains
    )
    size = min(int(args.train_subset_size), len(train_gt))
    indices = np.sort(
        np.random.default_rng(TRAIN_SUBSET_SEED).choice(
            len(train_gt), size=size, replace=False
        )
    ).tolist()
    train_ds = AttributeDataset(
        train_gt, config.data.root, transform, attribute_names=attribute_names
    )
    train_loader = _make_loader(
        Subset(train_ds, indices),
        batch_size=ctx.batch_size,
        workers=workers,
        device=device,
    )
    train_ids = [str(train_gt.iloc[index]["image"]) for index in indices]
    _dump_loader(
        model=model,
        loader=train_loader,
        device=device,
        out_dir=out_dir,
        split="train_subset",
        image_ids=train_ids,
    )
    print(f"Wrote artifacts under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
