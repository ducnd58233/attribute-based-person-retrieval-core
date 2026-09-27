# Agent instructions — attribute-based-person-retrieval-core

Solution repo for **WACV 2027 UPAR Challenge Track 2 (Attribute-based Person
Retrieval)**. Mounted as a git submodule of the docs-only
`attribute-based-person-retrieval-workspace` repo — read that repo's
`PROBLEM-DETAILS.md` for the competition rules and `RESEARCH-*.md`/`PLAN-*.md`
for the approach before writing any model code here.

<scope>

This repo holds training code, inference code, and the environment/tooling
needed to build a Codabench submission. It does **not** hold competition
rules or research notes — those live in the parent workspace and are not
duplicated here; link to them instead.

Two environments exist here, deliberately different:

- **Dev/training** (`environment.yml` + `pyproject.toml`, this repo's root):
  Python 3.13, CUDA 13.0, torch built against cu130 — whatever is fastest
  for training on your own hardware. Nothing here has to match the
  competition's runtime.
- **Submission/evaluation** (`deployments/docker/`): a local mirror of the
  organizers' exact evaluation container
  (`pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`, Python 3.11, torch
  2.4.1). Only code and weights inside the submission zip run here — no
  network access, everything the model needs must be vendored into the zip.

Do not let the two blend: a dependency added to `pyproject.toml` for
training does not exist in the evaluation container unless it is also
vendored into the submission zip.

</scope>

<precedence>

The parent workspace's `AGENTS.md` governs the parent repo, not this one.
Competition rules are read from the parent's `PROBLEM-DETAILS.md` (do not
fork a copy here); everything else about how this repo is built is decided
here.

</precedence>

<always_on>

- **conda before uv (MUST):** `uv` itself, `cudnn`, and the CUDA toolkit come
  from the conda environment, not from uv. Every `uv` command in this repo
  assumes the conda env is already active:

  ```sh
  conda env create -f environment.yml   # once
  conda activate wacv                   # every new shell, before any uv command
  uv sync                               # now safe to run
  uv run pre-commit install             # once per checkout
  ```

  `.pre-commit-config.yaml`'s ruff/black hooks run as `uv run ruff`/`uv run
  black` (`language: system`, not a pre-commit-managed environment), so a
  commit made from a shell without the `wacv` env active fails the same way
  a bare `uv` command does.

  Running `uv` without an active `wacv` conda environment is a
  defect, not a style preference: `uv` may not even be on `PATH` yet, and
  any GPU extension that links against CUDA/cudnn will pick up the wrong
  (or no) toolkit. Never install `uv` a second way (pipx, a system package
  manager) to work around this — activate conda instead.
- **Submission packaging (MUST):** before zipping a submission, build and
  run `deployments/docker/docker-compose.yml`'s `submission-test` service
  against it. A dependency that works in the dev conda/uv environment but
  is missing from the zip fails silently until the real evaluation run.
- **Never commit competition data.** Sample/dev copies of the datasets go
  under a gitignored `data/` (already in `.gitignore`); the real training
  data and hidden test set are never committed here, per the competition's
  licence terms in the parent workspace's `PROBLEM-DETAILS.md`.
- **Never commit trained model weights** to git history directly; use
  Git LFS, Hugging Face Hub, or another artifact store, and document
  wherever weights are pulled from in the model's own README.
- **Do not add automated tests to this repository unless the user explicitly
  asks for them.** Validate implementation work with the existing checks,
  static analysis, and bounded smoke runs instead.
- **No lint / type suppresses (MUST):** do not add `# noqa`, `# ruff: noqa`,
  `# type: ignore`, `# pyright: ignore`, or widen
  `pyproject.toml` ignore / per-file-ignore lists to land a change. Fix the
  code (or the typed API) so the existing rules pass. Prefer deferred
  `src.wacv` imports inside functions in `scripts/` over path-hack `# noqa`.
- **No unrequested code (MUST):** only add files and logic the current task
  acceptance criteria require. Prefer a script under `scripts/` over new
  packages, fixtures, or README sections. Delete scaffolding that is not
  needed for train / evaluate / package / submit.
- **No decorative runtime strings (MUST):** do not add print/log messages
  whose only job is to narrate that a check "passed cleanly" or similar
  fluff (e.g. `run.py imports cleanly with no network access`). Docker /
  CLI entrypoints should call the real API (`import run; run.load_model()`)
  and let failures surface as exceptions.
- **Optional accelerators stay optional (MUST):** Colab/Kaggle TPU (and any
  similar hosted accelerator) is a *runtime* concern, not a local-env pin.
  Do **not** add `torch_xla` (or equivalent) to `environment.yml` /
  `pyproject.toml` for the CUDA training machine. Detect/load with
  `importlib` inside the rare path only so importing `src.wacv` still works
  without that package. Keep `resolve_device` and the XLA train-step branch
  short — no multi-probe device inventories, no long install-essay
  `RuntimeError` strings, no helper sprawl for a one-line `xm.optimizer_step`.
  AMP / `GradScaler` remain CUDA-only; never enable them on XLA or CPU.
- **Colab/Kaggle notebook viz (MUST):** `scripts/export-colab-notebook.py`
  emits **one** Visualize stage after Eval (val split), not a second copy
  after Train. Attribute comparison figures must be readable in a notebook
  (dedicated text panel / large enough type with explicit TP/FP/FN meaning)
  — not a dense grid of 7pt xlabel dumps. Regenerate the gitignored
  notebook via the exporter; do not hand-edit the `.ipynb` as source of
  truth.
- **`Path(__file__)` in the Colab export — know which case you're in
  (MUST):** `scripts/export-colab-notebook.py` inlines each pipeline stage
  script's *body* directly as notebook-cell source, not loaded from a real
  file — that top-level code genuinely has no `__file__` in Colab, which is
  why the exporter rewrites `ROOT = Path(__file__).resolve().parents[1]` to
  `ROOT = Path.cwd()` specifically in those cells (`FILE_ROOT_RE`). A
  `src/wacv/*.py` **library module** is a different case: the exporter
  `%%writefile`'s it to real disk and it is then loaded via a normal
  `import` statement, so Python's import system sets `__file__` correctly —
  `Path(__file__)` is safe there (already used this way in
  `config.py::load_config`'s default root; verified empirically that
  `%%writefile`-then-`import` preserves `__file__`, both locally and by the
  fact this exact pattern already runs successfully in real Colab training
  runs). Don't "fix" a working `Path(__file__)` in a library module by
  assuming every `__file__` use is broken under Colab, and don't add
  `Path(__file__)` to a *stage script*'s top level. When in doubt, check
  which mechanism applies (inlined script body vs. `%%writefile`+`import`)
  before changing either direction.
- **Remote GPU training: Kaggle first, Colab second (MUST):** headless
  `Download → Train → Eval → Visualize → Package` runs on hosted GPUs go
  through `scripts/kaggle_train.py` (primary) or `scripts/colab_train.py`
  (secondary), never a hand-written one-off. Both build on
  `scripts/remote_stage.py`, which reuses `export-colab-notebook.py`'s own
  `stages_for_config`/`visualize_stage`/`experiment_identity` so the
  notebook path and these two headless paths cannot silently drift apart.
  Neither script needs this private repo to be `git clone`d from inside the
  remote sandbox — both stage a self-contained folder (this repo's
  `scripts/` + `src/wacv/` + one experiment YAML, never `data/`) and ship
  that instead.

  **Kaggle** (primary — cross-platform including native Windows; a pushed
  kernel runs server-side, so it survives a closed laptop):

  ```sh
  uv sync                              # pulls `kaggle` from the dev group
  # one-time auth: https://www.kaggle.com/settings -> Create New Token,
  # save the download as ~/.kaggle/kaggle.json (never commit it)
  uv run python scripts/kaggle_train.py --config configs/<name>.yml
  ```

  This pushes a `script`-type kernel (`enable_gpu`, `enable_internet` both
  on), polls `kaggle kernels status` until it reaches a terminal state (no
  local timeout unless `--timeout-hours` is passed — closing this terminal
  does not stop the kernel), then runs `kaggle kernels output` and copies
  the result into local `runs/<run_name>/`. `--stage-only` builds the
  staging folder and exits without needing the `kaggle` CLI, credentials,
  or network access at all.

  **Colab** (secondary — `google-colab-cli`, `uv tool install
  google-colab-cli`, supports **Linux/macOS only**; run from WSL2 on
  Windows, or use Kaggle instead). Unlike Kaggle, `colab exec` blocks this
  process in the foreground for the whole run — the local machine must stay
  up (use `tmux`/`nohup` for a run that must outlive a closed terminal):

  ```sh
  uv tool install google-colab-cli     # once per machine, not a project dep
  uv run python scripts/colab_train.py --config configs/<name>.yml --gpu T4
  ```

  This runs `colab new → upload → exec → download → stop` (`stop` always
  runs, even on failure, so a failed run doesn't leave a billed GPU session
  up). `--stage-only` works the same as Kaggle's, needing no `colab` CLI or
  Google auth.

  Both scripts read credentials only from each CLI's own standard location
  (Kaggle: the home-directory files *~/.kaggle/kaggle.json* or
  *~/.kaggle/access_token*, or `KAGGLE_USERNAME`/`KAGGLE_KEY`; Colab: its
  own OAuth/ADC flow) — never from a repo file, and never printed.
- **Notebook and remote-path check, after touching training code (MUST):**
  any change to `scripts/*.py` or `src/wacv/**` regenerates and
  re-validates every path that ships that code before the change counts as
  done:

  ```sh
  uv run python scripts/export-colab-notebook.py --config configs/<touched-experiment>.yml
  uv run python scripts/kaggle_train.py --config configs/<touched-experiment>.yml --stage-only
  uv run python scripts/colab_train.py  --config configs/<touched-experiment>.yml --stage-only
  ```

  All three fail loudly (non-zero exit) on a syntax error or an unresolved
  `src.wacv` import anywhere in the reachable pipeline — `--stage-only`
  needs no Kaggle/Colab credentials, so there is no excuse to skip it. When
  the change touches shared code (`src/wacv/engine.py`, `data.py`, and
  similar) rather than one experiment, run the three commands against
  `configs/smoke.yml` instead of picking one experiment arbitrarily. This is
  a syntax/import check, not a substitute for the deeper, already-documented
  bounded smoke run (`uv run python scripts/train.py --config
  configs/smoke.yml`) when the change might affect runtime behavior, not
  just imports.

</always_on>

<delivery_gates>

**Always**

- Keep the dev environment (`environment.yml`, `pyproject.toml`) and the
  submission environment (`deployments/docker/`) explicitly separate; never
  make the submission Dockerfile depend on anything from the dev
  environment being present.
- Test a submission inside `deployments/docker/` before uploading it to
  Codabench.

**Ask first**

- Any dependency that is not obviously needed for pedestrian attribute
  recognition or attribute-based retrieval (this repo currently trims
  anything unrelated, e.g. 3D-reconstruction tooling, from the dev
  environment — keep it that way; don't reintroduce broad "just in case"
  dependencies).
- Changing the pinned base image in `deployments/docker/Dockerfile` — it
  must track whatever Codabench's Submission page says, not be upgraded
  independently.

**Never**

- Commit competition data or trained weights to this repo.
- Use the hidden test set for anything (training, pseudo-labelling, manual
  inspection, hyperparameter selection) — disqualifying per the competition
  rules.
- Add code to `run.py`/the submission entry point that reaches the network
  at inference time.
- Commit Kaggle/Colab credentials (`kaggle.json`, `access_token`, OAuth
  config) anywhere in this repo, or `git clone` this repo from inside a
  Kaggle kernel or Colab VM (stage a self-contained folder instead —
  `scripts/remote_stage.py`).

</delivery_gates>
