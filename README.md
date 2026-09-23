# attribute-based-person-retrieval-core

Solution code for **WACV 2027 UPAR Challenge Track 2 (Attribute-based Person
Retrieval)**. Competition rules, research, and plan live in the parent
workspace (`attribute-based-person-retrieval-workspace`, this repo's
superproject) — start there, not here, for context.

## Setup
- Setup environment
```sh
conda env create -f environment.yml
conda activate wacv            # required before every uv command, see AGENTS.md
uv sync
uv run pre-commit install      # once per checkout
```

- Setup datasets
```sh
uv run scripts/download-dataset.py
```

## Training and evaluation

Experiments are config-driven (`configs/*.yml`, validated by
`src/wacv/config.py`); a smoke config (`configs/smoke.yml`, 1 epoch on a
64-image slice) exists to check the pipeline runs before committing to a
real training config like `configs/convnext_baseline.yml`.

```sh
uv run python scripts/train.py --config configs/smoke.yml
uv run python scripts/evaluate.py --config configs/smoke.yml \
    --checkpoint runs/smoke/checkpoints/best.pt
```

Each run writes to `run.output_dir/run.name/` (default `runs/<name>/`):
`config.json` (the resolved config), `metrics.jsonl`, `run.log`, and
`checkpoints/{last,best}.pt`. Set `run.resume` to a checkpoint path to
continue training from it.

## Testing a submission before upload

```sh
docker compose -f deployments/docker/docker-compose.yml run --rm submission-test
```

See `deployments/docker/docker-compose.yml` for how to point it at the
submission folder you're about to zip.
