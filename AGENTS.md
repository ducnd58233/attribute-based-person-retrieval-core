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
- **A function signature over `max-args` takes an existing config object,
  not a suppress (MUST):** when extracting a shared helper pushes a
  signature past `pyproject.toml`'s `[tool.ruff.lint.pylint] max-args`
  limit, look for a config object the callers already have in hand
  (`ExperimentConfig`, `RankingConfig`, ...) before adding parameters one
  at a time. This actually happened extracting `compute_ranking_metrics`
  into `src/wacv/evaluation/__init__.py`: three scalar parameters
  (`distance`, `temperature`, `clear_fusion`) pushed it to 9 args and
  failed `PLR0913`; both call sites already had the `RankingConfig` those
  three fields live on, so the fix was accepting that object instead of
  suppressing the rule or leaving the extraction unwritten.
- **`scripts/` file naming: `-`, never `_` (MUST):** every file directly
  under `scripts/` uses hyphens (`export-colab-notebook.py`,
  `sweep-ranking-distance.py`), matching the CLI-entrypoint convention
  already established there. This does not apply to `src/wacv/**` -
  Python package/module names there must stay valid identifiers
  (underscores), and cannot use hyphens at all.
- **`src/wacv/` folder layout by concern (MUST):** a concern gets its own
  subpackage once it is more than "one obvious file" - `src/wacv/models/`
  (network architectures), `src/wacv/evaluation/` (prediction, ranking
  distances, metric computation), `src/wacv/monitoring/` (`metrics.py`
  scoring functions, `tracking.py` `RunTracker` persistence/logging),
  `src/wacv/remote/` (hosted-GPU training backends). A single-file module
  that does not yet belong to one of these concerns (`data.py`,
  `config.py`, `checkpoint.py`, `losses.py`, `attributes.py`,
  `query_adapter.py`, `visualize.py`, `engine.py`) stays
  flat rather than being forced into a one-file package - do not create a
  subpackage without a second file that actually belongs in it.
- **No unrequested code (MUST):** only add files and logic the current task
  acceptance criteria require. Prefer a script under `scripts/` over new
  packages, fixtures, or README sections. Delete scaffolding that is not
  needed for train / evaluate / package / submit.
- **A well-known problem gets a well-known, uv-installable, cross-platform
  (Windows + Linux) library before it gets a hand-rolled reimplementation
  (MUST):** `src/wacv/remote/env.py`'s `load_dotenv` used to be a from-scratch
  `KEY=VALUE` parser with a comment justifying it ("not python-dotenv...
  not worth a dependency for") - that judgment call is reversed now:
  `python-dotenv` is official, pure Python (installs identically on
  Windows and Linux via `uv`, no compiled extension), and its own
  `override=False` already gets the "real env var wins" behavior right,
  so the hand-rolled parser bought nothing except a second copy of logic
  to maintain and its own edge cases (quoting, comments, blank lines) to
  keep in sync with the real spec. Reach for the real library first;
  reimplement only when nothing free-and-installable actually covers the
  need. This is a case-by-case judgment, not a blanket "always add a
  dependency": `src/wacv/remote/kaggle.py`'s `wait_for_terminal_status`
  open-ended poll-until-terminal-or-timeout loop stays hand-rolled on
  purpose - it is domain logic (when is this specific Kaggle kernel
  actually done), not a generic single-call retry a library like
  `tenacity` is built for, and reaching for one there would add a
  dependency for something already minimal and correct.
- **Grep the whole repo before deleting a function or class as dead
  (MUST):** "looks unused in this file" is not evidence. Search every
  `src/`, `scripts/`, `tests/`, and `submission/` file for the exact name
  before removing it, and note in the PR description (or commit body)
  that the search came back empty. `data.py::load_query_ids` and
  `query_adapter.py::unwrap_backbone` were removed this way - a
  same-file-only look would have missed that `scripts/export-colab-
  notebook.py`'s AST-walk-based inlining is also a real caller for some
  functions, which a narrower search would not have ruled out.
- **The same predict/rank/score (or similar) sequence in two files gets
  extracted immediately, not on a later cleanup pass (MUST):** if writing
  a second copy of logic that already exists elsewhere in the codebase,
  stop and extract a shared function in the same change, into whichever
  module already owns that concern (see the folder-layout-by-concern rule
  above). `Trainer._ranking_metrics` (`engine.py`) and `scripts/
  evaluate.py`'s `_evaluate_checkpoint` independently reimplemented the
  same predict → rank → score sequence across several earlier PRs before
  one refactor finally noticed and extracted `compute_ranking_metrics` -
  the second copy should never have been written standalone.
- **A memory fix for a ranking function is not done until the whole
  ranking stage is profiled at real gallery scale, not just the one
  function that was touched (MUST):** Codabench submission 948099 was
  "fixed" by making `_dual_softmax_rerank` chunked float32 (see the
  dtype-mirroring rule above); the very next submission (948604), same
  zip pattern, failed with the identical opaque error. Profiling the
  *whole* ranking stage at the real E17 shape (Q=3462, G=33407) - not
  just the function that was already suspected - found the actual
  remaining peak was in `_hybrid_distance`/`_minmax_per_query`, never
  touched by the first fix. After any change meant to fix or bound a
  submission-side OOM, profile peak RSS across the full call it sits
  inside (`rank_gallery`, not just the one function), at the real
  gallery/query counts recorded in this submission's own
  `records/submissions/*/SUMMARY.md` - a unit test on a small array
  proves the formula, never the memory footprint.
- **Prefer numpy's in-place ops (`+=`, `out=`, mutate-and-return) over the
  naive expression form for any O(Q×G) array on the submission's ranking
  path (MUST):** `a + b - 2*c`, `np.where(cond, x, np.zeros_like(y))`, and
  `weight * a + (1 - weight) * b` each read as a single line but allocate
  one full-size temporary per operator - on a ~3462×33407 gallery that is
  ~463MB per float32 temporary, and several stack up across one
  expression. `_hybrid_distance`, `_minmax_per_query`, `_nll_distance`,
  and `_l1_distance` (and their identical library copies in
  `src/wacv/evaluation/__init__.py`) held up to four such arrays alive at
  once before this was rewritten to mutate a single scratch buffer
  through each step - cut measured peak RSS on the real shape by 47%
  (2808MB -> 1492MB) with zero output change (verified against this
  repo's own naive-loop parity tests, not assumed). Apply this only where
  the array being mutated is a scratch value nothing else reads
  afterward - state that ownership assumption in the function's
  docstring, the way `_dual_softmax_rerank`'s and `_minmax_per_query`'s
  now do, so a future caller does not get a silently mutated argument.
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
- **Moving a file changes its `Path(__file__).resolve().parents[N]` index
  (MUST):** the index counts directories between the file and the repo
  root, so moving a file one level deeper (e.g. into a new subpackage)
  means every `parents[N]` call in it needs `N + 1`, not the same number.
  Nothing in ruff/pyright/black catches a now-wrong-but-still-valid index;
  check every file at the *new* depth for what index they already use
  (`src/wacv/remote/stage.py` uses `parents[3]`) before assuming the old
  number still applies. Moving `tracking.py` into `src/wacv/monitoring/`
  left `parents[2]` silently resolving to the wrong root until this was
  caught by that cross-check, not by any tool.
- **Remote GPU training: one entrypoint, one backend per hosted provider
  (MUST):** headless `Download → Train → Eval → Visualize → Package` runs
  on hosted GPUs go through `scripts/remote-training.py <backend> --config
  ...`, never a hand-written one-off. `<backend>` is `kaggle` (primary —
  cross-platform including native Windows; a pushed kernel runs
  server-side, so it survives a closed laptop) or `colab` (secondary —
  `google-colab-cli`, **Linux/macOS only**, run from WSL2 on Windows;
  `colab exec` blocks the local process for the whole run, so use
  `tmux`/`nohup` for one that must outlive a closed terminal, or prefer
  Kaggle). Add a third hosted provider by dropping a module under
  `src/wacv/remote/` that exposes a `register(subparsers)` function (same
  shape as `kaggle.py`/`colab.py`) and listing it in
  `scripts/remote-training.py`'s `BACKENDS`; do not add a second top-level
  script.

  Every backend builds on `src/wacv/remote/stage.py`, which reuses
  `scripts/export-colab-notebook.py`'s own
  `stages_for_config`/`visualize_stage`/`experiment_identity` so the
  notebook path and every headless backend stay in sync. None of them need
  this private repo to be `git clone`d from inside the remote sandbox —
  each stages a self-contained folder (this repo's `scripts/` + `src/wacv/`
  + one experiment YAML, never `data/`) and ships that instead.

  ```sh
  uv sync                                        # pulls kaggle+packaging from dev
  cp .env.example .env                           # fill in KAGGLE_USERNAME/KAGGLE_KEY
  # https://www.kaggle.com/settings -> Create New Token gives both
  uv run python scripts/remote-training.py kaggle --config configs/<name>.yml

  uv tool install google-colab-cli               # once per machine, not a project dep
  uv run python scripts/remote-training.py colab --config configs/<name>.yml --gpu T4
  ```

  `remote-training.py` loads `.env` (gitignored; copy from `.env.example`,
  never commit it) into the environment at startup, without overriding a
  real env var already set — `KAGGLE_USERNAME`/`KAGGLE_KEY` reach the
  `kaggle` CLI subprocess that way, and this repo's own code never reads
  the key value, only ever `username` when falling back to the
  home-directory file *~/.kaggle/kaggle.json*.

  **Colab with no human present (verified):** `colab`'s default
  `--auth adc` uses Google's Application Default Credentials. A human
  runs, once, on any machine with a browser (does not have to be the
  agent's machine): `gcloud auth application-default login
  --no-launch-browser` — this prints a URL plus a device code, completes
  consent wherever the browser is, and writes a JSON credentials file to
  a well-known path (*~/.config/gcloud/application_default_credentials.json*
  on Linux/macOS). Every later `colab` invocation reuses that file
  automatically, with zero further interaction — this is ADC's documented
  contract, not something this repo adds. Point `GOOGLE_APPLICATION_
  CREDENTIALS` in `.env` at a copy of that file and no code change is
  needed: `env.py::load_dotenv` forwards it into `os.environ` like any
  other `.env` line, and `subprocess.run` already inherits the parent
  environment. **`UNVERIFIED`:** whether Colab's own runtime-provisioning
  quota accepts a service-account identity — only tested against a
  personal Google account's own ADC file; use the human's own account's
  credentials, not a separate service account, until that is confirmed.

  `--stage-only` on either backend builds the staging folder and exits
  without needing that backend's CLI, credentials, or network access at all. The Kaggle backend pushes a `script`-type
  kernel (`enable_gpu`, `enable_internet` both on) and polls `kaggle
  kernels status` until terminal (no local timeout unless `--timeout-hours`
  is passed), then `kaggle kernels output`s into local `runs/<run_name>/`.
  The Colab backend runs `colab new → upload → exec → download → stop`
  (`stop` always runs, even on failure, so a failed run doesn't leave a
  billed GPU session up).

  **Staged experiment configs must stay secret-free** (no API keys, W&B
  tokens, or similar in a `configs/*.yml`) — the staging folder ships that
  YAML verbatim to a third-party sandbox (Kaggle/Colab), and nothing here
  scans it for embedded secrets.
- **Notebook and remote-path check, after touching training code (MUST):**
  any change to `scripts/*.py` or `src/wacv/**` regenerates and
  re-validates every path that ships that code before the change counts as
  done:

  ```sh
  uv run python scripts/export-colab-notebook.py --config configs/<touched-experiment>.yml
  uv run python scripts/remote-training.py kaggle --config configs/<touched-experiment>.yml --stage-only
  uv run python scripts/remote-training.py colab  --config configs/<touched-experiment>.yml --stage-only
  ```

  All three fail loudly (non-zero exit) on a syntax error or an unresolved
  `src.wacv` import anywhere in the reachable pipeline, and `--stage-only`
  also compiles the generated `run_remote.py` entrypoint itself before
  writing it — needs no Kaggle/Colab credentials, so there is no excuse to
  skip it. When the change touches shared code (`src/wacv/engine.py`,
  `data.py`, and similar) rather than one experiment, run the three
  commands against `configs/smoke.yml` instead of picking one experiment
  arbitrarily. This is a syntax/import check, not a substitute for the
  deeper, already-documented bounded smoke run (`uv run python
  scripts/train.py --config configs/smoke.yml`) when the change might
  affect runtime behavior, not just imports.

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
- Commit Kaggle/Colab credentials (`.env`, `kaggle.json`, `access_token`,
  OAuth config) anywhere in this repo, or `git clone` this repo from inside
  a Kaggle kernel or Colab VM (stage a self-contained folder instead —
  `src/wacv/remote/stage.py`).

</delivery_gates>
