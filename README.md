# attribute-based-person-retrieval-core

Solution code for **WACV 2027 UPAR Challenge Track 2 (Attribute-based Person
Retrieval)**. Competition rules, research, and plan live in the parent
workspace (`attribute-based-person-retrieval-workspace`, this repo's
superproject) — start there, not here, for context.

## Setup

Activate the conda env **before every `uv` command** (see `AGENTS.md`):

```sh
conda env create -f environment.yml   # once
conda activate wacv
uv sync
```

## Scripts (`uv run`)

All commands below assume `conda activate wacv` and the repo root as cwd.

Typical local pipeline:

```sh
uv run python scripts/download-dataset.py
uv run python scripts/train.py --config configs/convnext_baseline.yml
uv run python scripts/evaluate.py --config configs/convnext_baseline.yml
uv run python scripts/visualize.py --config configs/convnext_baseline.yml --split val --limit 4
uv run python scripts/package-submission.py \
  --checkpoint runs/convnext-baseline/checkpoints/best.pt \
  --config configs/convnext_baseline.yml
```

Colab notebook export (artifact under `notebooks/`, not committed):

```sh
uv run python scripts/export-colab-notebook.py --config configs/convnext_baseline.yml
```

### `scripts/download-dataset.py`

Clone the organizers' starter kit (cached), copy annotations, download
Market1501 / PA100K / PETA into `data/`.

| Argument | Required | Default | Description |
|---|---|---|---|
| *(none)* | — | — | No CLI flags; paths are fixed (`data/`, `.cache/UPAR-Challenge-2027/`). |

```sh
uv run python scripts/download-dataset.py
```

### `scripts/train.py`

Train an attribute classifier from a validated experiment YAML.

| Argument | Required | Default | Description |
|---|---|---|---|
| `--config` | no | `configs/convnext_baseline.yml` | Experiment YAML (`model` / `data` / `optimization` / `run`). |

```sh
uv run python scripts/train.py
uv run python scripts/train.py --config configs/convnext_baseline.yml
```

Checkpoints land under `runs/<run.name>/checkpoints/` (`best.pt`, `last.pt`),
using `run.name` / `run.output_dir` from the YAML.

### `scripts/evaluate.py`

Evaluate a checkpoint on the Track 2 validation split (ranking + mA).

| Argument | Required | Default | Description |
|---|---|---|---|
| `--config` | no | `configs/convnext_baseline.yml` | Experiment YAML (data paths, model, device). |
| `--checkpoint` | no | `runs/<run.name>/checkpoints/best.pt` | Weights to load; derived from the config when omitted. |

```sh
uv run python scripts/evaluate.py --config configs/convnext_baseline.yml
uv run python scripts/evaluate.py \
  --config configs/convnext_baseline.yml \
  --checkpoint runs/convnext-baseline/checkpoints/last.pt
```

### `scripts/visualize.py`

Plot ground-truth attributes vs predictions for the first N samples of a
task1 split.

| Argument | Required | Default | Description |
|---|---|---|---|
| `--config` | no | `configs/convnext_baseline.yml` | Experiment YAML. |
| `--checkpoint` | no | `runs/<run.name>/checkpoints/best.pt` | Weights to load. |
| `--split` | no | `val` | `train` or `val` (task1). |
| `--limit` | no | `4` | Number of leading samples to score and plot. |
| `--threshold` | no | `0.5` | Probability cutoff for a positive attribute. |
| `--out` | no | *(unset)* | Optional path to save the comparison figure. |
| `--no-show` | no | off | Skip interactive `plt.show` (still saves when `--out` is set). |

```sh
uv run python scripts/visualize.py --config configs/convnext_baseline.yml
uv run python scripts/visualize.py \
  --config configs/convnext_baseline.yml \
  --split train \
  --limit 8 \
  --threshold 0.4 \
  --out runs/convnext-baseline/viz_train.png \
  --no-show
```

### `scripts/package-submission.py`

Assemble `build/submission/` from the tracked template + a training checkpoint
(clean `assets/model.pt` + `model_card.json`).

| Argument | Required | Default | Description |
|---|---|---|---|
| `--checkpoint` | **yes** | — | Trainer checkpoint (`best.pt` / `last.pt` or a state dict). |
| `--config` | no | `configs/convnext_baseline.yml` | Rebuild architecture for a clean weight export. |
| `--out` | no | `build/submission` | Destination folder (contents are replaced). |
| `--template` | no | `submission` | Tracked `run.py` + `metadata.yaml` template directory. |

```sh
uv run python scripts/package-submission.py \
  --checkpoint runs/convnext-baseline/checkpoints/best.pt

uv run python scripts/package-submission.py \
  --checkpoint runs/convnext-baseline/checkpoints/best.pt \
  --config configs/convnext_baseline.yml \
  --out build/submission \
  --template submission
```

### `scripts/export-colab-notebook.py`

Generate a Colab/Kaggle notebook from `scripts/` + collected `src/wacv`
modules and the chosen experiment YAML.

| Argument | Required | Default | Description |
|---|---|---|---|
| `--config` | no | `configs/convnext_baseline.yml` | Experiment YAML baked into train / eval / package / persist cells. |
| `--out` | no* | `notebooks/train_pipeline.ipynb` | Full destination `.ipynb` path. |
| `--name` | no* | — | Basename under `notebooks/` (adds `.ipynb` if missing). |

\* `--out` and `--name` are mutually exclusive. If neither is set, the default
`--out` path above is used.

```sh
# default → notebooks/train_pipeline.ipynb
uv run python scripts/export-colab-notebook.py --config configs/convnext_baseline.yml

# name only → notebooks/convnext_baseline.ipynb
uv run python scripts/export-colab-notebook.py \
  --config configs/convnext_baseline.yml \
  --name convnext_baseline

# full path
uv run python scripts/export-colab-notebook.py \
  --config configs/convnext_baseline.yml \
  --out notebooks/custom.ipynb
```

## Testing a submission before upload

```sh
docker compose -f deployments/docker/docker-compose.yml run --rm submission-test
```

See `deployments/docker/docker-compose.yml` for how to point it at the
submission folder you're about to zip.
