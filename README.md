# attribute-based-person-retrieval-core

Solution code for **WACV 2027 UPAR Challenge Track 2 (Attribute-based Person
Retrieval)**. Competition rules, research, and plan live in the parent
workspace (`attribute-based-person-retrieval-workspace`, this repo's
superproject) — start there, not here, for context.

## Setup

```sh
conda env create -f environment.yml
conda activate wacv            # required before every uv command, see AGENTS.md
uv sync
uv run pre-commit install      # once per checkout
```

## Layout

```text
deployments/docker/   local mirror of the competition's evaluation container,
                       for testing a submission zip before upload
environment.yml         conda: system-level CUDA/cudnn + uv itself
pyproject.toml          uv: Python dependencies for training/dev
.pre-commit-config.yaml ruff/black + file hygiene, run via `uv run` (language: system)
AGENTS.md               rules for agents/contributors working in this repo
```

## Testing a submission before upload

```sh
docker compose -f deployments/docker/docker-compose.yml run --rm submission-test
```

See `deployments/docker/docker-compose.yml` for how to point it at the
submission folder you're about to zip.
