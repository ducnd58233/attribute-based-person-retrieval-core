# attribute-based-person-retrieval-core

Solution code for **WACV 2027 UPAR Challenge Track 2 (Attribute-based Person
Retrieval)**.

**To reproduce the final entry** (Codabench submission 995287: download the
scored package, or retrain its nine members and rebuild it), follow
[`REPRODUCE.md`](REPRODUCE.md).

Everything needed to understand and reproduce the entry is in this repository:
`REPRODUCE.md`, the configs under `configs/`, and the code. `AGENTS.md`,
`CLAUDE.md` and `CURSOR.md` are instructions for the coding assistants used
during development; they refer to private planning notes and are not needed to
run anything here.

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

### `scripts/download-hf-model.py`

Download a Hugging Face model repo once into `runs/hf/models/<name>`
(skipped if already complete). Loads `.env` at startup for Hub auth.

```sh
cp .env.example .env   # set HF_TOKEN=... (never commit); or use hf auth login
uv run python scripts/download-hf-model.py google/embeddinggemma-300m --agree-gated-terms
```

Gated repos need a Hub licence accept plus `HF_TOKEN` in `.env` (or
`HUGGING_FACE_HUB_TOKEN`, or interactive `hf auth login` as fallback) and
`--agree-gated-terms`.

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
| `--tta` | no | keep config | Override packaged `inference_tta` views (e.g. `identity`) without editing the train YAML. |

```sh
uv run python scripts/package-submission.py \
  --checkpoint runs/convnext-baseline/checkpoints/best.pt

uv run python scripts/package-submission.py \
  --checkpoint runs/convnext-baseline/checkpoints/best.pt \
  --config configs/convnext_baseline.yml \
  --out build/submission \
  --template submission

# Heavy backbones: ship identity-only TTA if multi-view overruns the 1h wall clock
uv run python scripts/package-submission.py \
  --checkpoint runs/e17_clip_b16_lpft/checkpoints/best.pt \
  --config configs/e17_clip_b16_lpft.yml \
  --tta identity
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

### `scripts/remote-training.py`

Push the `Download → Train → Eval → Visualize → Package` pipeline to a
hosted GPU (Kaggle kernel or Colab session), wait for it, and pull the
result into local `runs/<run.name>/` — for training on a machine other than
this one. Backends live under `src/wacv/remote/`; `kaggle` is the primary,
cross-platform (including Windows) one since a pushed kernel keeps running
on Kaggle's servers even if this machine goes to sleep, `colab` is
secondary and Linux/macOS-only (`google-colab-cli`, `uv tool install
google-colab-cli`; run it from WSL2 on Windows). See `AGENTS.md` "Remote GPU
training" for the full walkthrough and the credential/security rules.

```sh
cp .env.example .env   # fill in KAGGLE_* / HF_TOKEN as needed; never commit
uv run python scripts/remote-training.py kaggle --config configs/convnext_baseline.yml
uv run python scripts/remote-training.py colab  --config configs/convnext_baseline.yml --gpu T4
```

| Argument (both backends) | Required | Default | Description |
|---|---|---|---|
| `--config` | **yes** | — | Experiment YAML to run remotely. |
| `--stage-only` | no | off | Build the staging folder and exit; no backend CLI or credentials needed. |

| `kaggle`-only argument | Required | Default | Description |
|---|---|---|---|
| `--accelerator` | no | `NvidiaTeslaT4` | Kaggle GPU/TPU accelerator id. |
| `--kaggle-username` | no | from `.env` or the home-directory *~/.kaggle/kaggle.json* | Overrides the resolved Kaggle username. |
| `--poll-interval` | no | `60` | Seconds between `kaggle kernels status` checks. |
| `--timeout-hours` | no | unlimited | Give up *polling* after this long; the kernel itself keeps running on Kaggle regardless. |

| `colab`-only argument | Required | Default | Description |
|---|---|---|---|
| `--gpu` | no | `T4` | Colab GPU type. |
| `--session` | no | `wacv-train` | `google-colab-cli` session name. |

#### Credentials (one-time setup)

**Hugging Face Hub** (gated models such as EmbeddingGemma): create a read
token at https://huggingface.co/settings/tokens, accept the model licence
on the repo page, and set `HF_TOKEN` in `.env` (see `.env.example`).
`HUGGING_FACE_HUB_TOKEN` is an accepted alias. Interactive
`hf auth login` / `huggingface-cli login` remains an optional fallback.
`scripts/download-hf-model.py` loads `.env` automatically.

**Kaggle:** go to https://www.kaggle.com/settings → "Create New Token" →
downloads `kaggle.json` with your `username` and `key`. Either paste both
into `.env` as `KAGGLE_USERNAME` / `KAGGLE_KEY`, or drop the file at
*~/.kaggle/kaggle.json* (the CLI's own default location) — `.env` is
enough on its own, the file is only a fallback.

**Colab:** run once, on any machine with a browser (does not have to be
the machine that will run `remote-training.py`):

```sh
gcloud auth application-default login --no-launch-browser
```

This prints a URL and a device code; finish the consent flow wherever the
browser is, then paste the code back. It writes a credentials file to
*~/.config/gcloud/application_default_credentials.json* (Linux/macOS) or
*%APPDATA%\gcloud\application_default_credentials.json* (Windows). Copy
that file somewhere the training machine/agent can read it, then set
`GOOGLE_APPLICATION_CREDENTIALS` in `.env` to that path — every later
`colab` invocation reuses it automatically, with no further login. Use
your own personal Google account for this, not a service account
(`UNVERIFIED` whether Colab's runtime quota accepts one).

```sh
# check it can still run, no credentials needed
uv run python scripts/remote-training.py kaggle --config configs/smoke.yml --stage-only
uv run python scripts/remote-training.py colab  --config configs/smoke.yml --stage-only
```

## Testing a submission before upload

```sh
docker compose -f deployments/docker/docker-compose.yml run --rm submission-test
```

`load_model` only. For a Codabench-shaped `rank_gallery` smoke (any package —
ConvNeXt / CLIP / LoRA), point `SUBMISSION_DIR` / `DATA_DIR` at the folder you
are about to zip and the local data root, then:

```sh
HARNESS_N_GALLERY=2048 HARNESS_N_QUERIES=512 \
  docker compose -f deployments/docker/docker-compose.yml run --rm submission-rank-smoke
```

See `deployments/docker/docker-compose.yml` for how to point it at the
submission folder you're about to zip.
