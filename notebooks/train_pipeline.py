# %% [markdown]
# WACV Track 2 — train pipeline (Colab / Kaggle)
#
# Source of truth: this file (`notebooks/train_pipeline.py`). Regenerate the
# `.ipynb` with:
#
# ```text
# uv run python scripts/export_colab_notebook.py --check
# ```
#
# Cells contain the download / train / eval / package **orchestration** (same
# logic as `scripts/*.py`), calling into `src.wacv`. Do not paste a second
# `Trainer` here.
#
# **Secrets:** use the host secret store or environment variables — never
# hardcode tokens in this notebook.

# %% [markdown]
## Setup — locate repo root

# %%
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CANDIDATES = [
    Path(os.environ["REPO_ROOT"]) if os.environ.get("REPO_ROOT") else None,
    Path.cwd(),
    Path.cwd() / "attribute-based-person-retrieval-core",
    Path("/kaggle/working/attribute-based-person-retrieval-core"),
    Path("/content/attribute-based-person-retrieval-core"),
]


def find_repo() -> Path:
    for candidate in CANDIDATES:
        if candidate is None:
            continue
        if (candidate / "src" / "wacv").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the core repo (needs src/wacv/). " "Clone it or set REPO_ROOT."
    )


REPO = find_repo()
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA = REPO / "data"
CONFIG_PATH = REPO / "configs" / "convnext_baseline.yml"
print("REPO =", REPO)
print("GPU  =", subprocess.getoutput("nvidia-smi -L") or "none detected")

# %% [markdown]
## Setup — install Python deps
#
# Colab/Kaggle often already ship a CUDA torch build. Install the remaining
# project dependencies with `pip`.

# %%
import subprocess
import sys

pkgs = [
    "numpy",
    "pillow",
    "pandas",
    "tqdm",
    "pydantic",
    "pyyaml",
    "gdown",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

try:
    import torch

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except ImportError:
    print("torch missing — install a CUDA build matching this host, then re-run")

# %% [markdown]
## Download
#
# Starter-kit clone, annotation copy, image download, and PETA remap
# (same steps as `scripts/download-dataset.py`).

# %%
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

CACHE = REPO / ".cache" / "UPAR-Challenge-2027"
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
    result = subprocess.run(cmd, cwd=str(starter), check=False)
    if result.returncode != 0:
        print("BTC script exited non-zero (expected on Windows PETA remap)")
    remap_peta(starter)


starter = clone_starter()
copy_annotations(starter)
download_images(starter)
print(f"Done under {DATA}")

# %% [markdown]
## Train
#
# Full-data ConvNeXt-Base via `configs/convnext_baseline.yml` (same
# orchestration as `scripts/train.py`).

# %%
from __future__ import annotations

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.wacv.attributes import ATTRIBUTE_NAMES, NUM_ATTRIBUTES
from src.wacv.config import ExperimentConfig, load_config
from src.wacv.data import (
    IMAGE_TRANSFORM,
    TRAIN_TRANSFORM,
    AttributeDataset,
    load_gt,
)
from src.wacv.engine import Trainer, resolve_device, seed_everything
from src.wacv.models import build_model
from src.wacv.tracking import RunTracker


def build_loaders(
    config: ExperimentConfig, device: torch.device
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    annotation_root = config.data.root / "annotations" / "task1"
    train_gt = load_gt(annotation_root / "train" / "gt.csv")
    val_gt = load_gt(annotation_root / "val" / "gt.csv")
    if config.data.limit is not None:
        train_gt = train_gt.iloc[: config.data.limit]
        val_gt = val_gt.iloc[: config.data.limit]

    train_transform = (
        TRAIN_TRANSFORM if config.data.train_augmentation else IMAGE_TRANSFORM
    )
    train_dataset = AttributeDataset(train_gt, config.data.root, train_transform)
    val_dataset = AttributeDataset(val_gt, config.data.root, IMAGE_TRANSFORM)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
    )

    positive_rates = train_gt[list(ATTRIBUTE_NAMES)].astype(float).mean()
    positive_rates = positive_rates.clip(lower=1e-6)
    weights = ((1.0 - positive_rates) / positive_rates).clip(
        upper=config.optimization.pos_weight_clip
    )
    pos_weight = torch.tensor(weights.to_numpy(), dtype=torch.float32)
    return train_loader, val_loader, pos_weight


config = load_config(CONFIG_PATH)
seed_everything(config.run.seed)
device = resolve_device(config.run.device)
train_loader, val_loader, pos_weight = build_loaders(config, device)

model = build_model(config.model, num_attributes=NUM_ATTRIBUTES).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.optimization.learning_rate,
    weight_decay=config.optimization.weight_decay,
)
scheduler = ReduceLROnPlateau(
    optimizer,
    factor=config.optimization.scheduler_factor,
    patience=config.optimization.scheduler_patience,
)

with RunTracker(config) as tracker:
    tracker.logger.info("device=%s model=%s", device, config.model.name)
    result = Trainer(
        model, optimizer, scheduler, criterion, device, tracker, config
    ).fit(train_loader, val_loader)
    tracker.logger.info(
        "finished start_epoch=%d end_epoch=%d global_step=%d best_val_loss=%.6f",
        result.start_epoch,
        result.end_epoch,
        result.global_step,
        result.best_metric,
    )

CHECKPOINT = config.run.output_dir / config.run.name / "checkpoints" / "best.pt"
print("checkpoint =", CHECKPOINT)

# %% [markdown]
## Eval
#
# Track-2 val metrics (mADM / mAP / Rank-k / mINP / mA) — same path as
# `scripts/evaluate.py`.

# %%
from __future__ import annotations

import json

import numpy as np
from torch.utils.data import DataLoader

from src.wacv.attributes import ATTRIBUTE_NAMES, NUM_ATTRIBUTES
from src.wacv.checkpoint import load_model_weights
from src.wacv.config import load_config
from src.wacv.data import AttributeDataset, load_gt, load_queries
from src.wacv.engine import resolve_device
from src.wacv.evaluation import l1_distance, mean_accuracy, predict_probabilities
from src.wacv.metrics import evaluate_ranking
from src.wacv.models import build_model

config = load_config(CONFIG_PATH)
device = resolve_device(config.run.device)
checkpoint = config.run.output_dir / config.run.name / "checkpoints" / "best.pt"

val_root = config.data.root / "annotations" / "task2" / "val"
ground_truth = load_gt(val_root / "gt.csv")
queries = load_queries(val_root / "queries.csv")
if config.data.limit is not None:
    ground_truth = ground_truth.iloc[: config.data.limit]
    queries = queries[: config.data.limit]

loader = DataLoader(
    AttributeDataset(ground_truth, config.data.root),
    batch_size=config.data.batch_size,
    num_workers=config.data.num_workers,
    pin_memory=device.type == "cuda",
    persistent_workers=config.data.num_workers > 0,
)
inference_model = config.model.model_copy(update={"pretrained": False})
model = build_model(inference_model, num_attributes=NUM_ATTRIBUTES).to(device)
load_model_weights(checkpoint.resolve(), model, map_location=device)

probabilities = predict_probabilities(
    model, loader, device, progress=config.run.progress
)
attributes = ground_truth[list(ATTRIBUTE_NAMES)].to_numpy(dtype=np.int64)
metrics = evaluate_ranking(queries, attributes, l1_distance(queries, probabilities))
metrics["mA"] = mean_accuracy(probabilities, attributes)
serialized = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
run_dir = config.run.output_dir / config.run.name
if run_dir.is_dir():
    (run_dir / "evaluation.json").write_text(serialized, encoding="utf-8")
print(serialized, end="")

# %% [markdown]
## Package
#
# Assemble `build/submission/` from the tracked template + checkpoint, then zip
# for Codabench (same steps as `scripts/package-submission.py`).

# %%
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from src.wacv.attributes import NUM_ATTRIBUTES
from src.wacv.checkpoint import load_model_weights
from src.wacv.config import load_config
from src.wacv.models import build_model

config = load_config(CONFIG_PATH)
checkpoint = config.run.output_dir / config.run.name / "checkpoints" / "best.pt"
template = REPO / "submission"
out = REPO / "build" / "submission"
zip_path = REPO / "build" / "submission.zip"


def export_weights(
    checkpoint_path: Path, config_path: Path, destination: Path
) -> dict[str, Any]:
    cfg = load_config(config_path)
    model = build_model(
        cfg.model.model_copy(update={"pretrained": False}),
        num_attributes=NUM_ATTRIBUTES,
    )
    load_model_weights(
        checkpoint_path.resolve(), model, map_location=torch.device("cpu")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    return {
        "name": cfg.model.name,
        "dropout": cfg.model.dropout,
        "num_attributes": NUM_ATTRIBUTES,
        "source_checkpoint": str(checkpoint_path),
    }


run_py = template / "run.py"
metadata = template / "metadata.yaml"
if not run_py.is_file() or not metadata.is_file():
    raise FileNotFoundError(f"Template missing run.py/metadata.yaml under {template}")

if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
shutil.copy2(run_py, out / "run.py")
shutil.copy2(metadata, out / "metadata.yaml")

assets = out / "assets"
assets.mkdir()
card = export_weights(checkpoint, CONFIG_PATH, assets / "model.pt")
(assets / "model_card.json").write_text(
    json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=out)
print(json.dumps({"out": str(out), "zip": str(zip_path), "card": card}, indent=2))
print("upload the zip to Codabench; do not commit weights or data")
