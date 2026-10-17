# Agent instructions — attribute-based-person-retrieval-core

Solution repo for **WACV 2027 UPAR Challenge Track 2 (Attribute-based Person
Retrieval)**. Mounted as a git submodule of the docs-only
`attribute-based-person-retrieval-workspace` repo — read that repo's
`PROBLEM-DETAILS.md` for the competition rules and `RESEARCH-*.md`/`PLAN-*.md`
for the approach before writing any model code here.

**Re-read parent `PROBLEM-DETAILS.md` Terms/Conditions before research /
train / package / submit (MUST):** before starting or continuing training,
packaging, or Codabench submission work in this repo, re-read the **Terms
and Conditions** section in the parent workspace's `PROBLEM-DETAILS.md`
(including the note that the live Terms tab names Track 1). Apply the shared
constraints that clearly carry to Track 2: organizer training data only; no
external data to fine-tune for the task; ImageNet / foundation / VLM
pretrains OK if not specifically trained for PAR; no use of test images
(train, pseudo-label, annotate, hyperparam selection); submission must not
access the network or tamper with the evaluation environment; data licence /
research-only. Track 2 ranking metric and submission format come from Track
2 pages in that same file — do not invent Track-2-only term rewrites.

<scope>

This repo holds training code, inference code, and the environment/tooling
needed to build a Codabench submission. It does **not** hold competition
rules or research notes — those live in the parent workspace and are not
duplicated here; link to them instead.

Two environments exist here, deliberately different:

- **Dev/training** (`environment.yml` + `pyproject.toml`, this repo's root):
  whatever Python, CUDA and torch versions are fastest for training on your
  own hardware; the pinned versions are in those two files. Nothing here has
  to match the competition's runtime.
- **Submission/evaluation** (`deployments/docker/`): a local mirror of the
  organizers' exact evaluation container; the image is pinned in
  `deployments/docker/Dockerfile` and must match the parent workspace's
  `PROBLEM-DETAILS.md`. Only code and weights inside the submission zip run
  here — no network access, everything the model needs must be vendored
  into the zip.

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
- **Submission package zip names (MUST):** name the durable zip for the
  **experiment** (card id + short model/slug) in **snake_case**, matching
  scored style already on disk — e.g. `e101_teacher_identity.zip`,
  `e118_teacher_stronger_erasing.zip`, `e126_e107_e112_avg.zip`. Pass
  `scripts/package-submission.py --out build/<experiment_slug>` so the
  emitted zip is `build/<experiment_slug>.zip`, not the default
  `build/submission.zip`. KEEP copies under `build/_KEEP_<…>/` use the
  same experiment-named zip; never leave a durable KEEP artifact as
  generic `submission.zip`. Align the slug with the parent workspace's
  kebab-case record folder `<codabench-id>-<model-name>` when an ID
  exists. Do not rename an already-scored B* zip (e.g.
  `e101_teacher_identity.zip`).
- **Never commit competition data.** Sample/dev copies of the datasets go
  under a gitignored `data/` (already in `.gitignore`); the real training
  data and hidden test set are never committed here, per the competition's
  licence terms in the parent workspace's `PROBLEM-DETAILS.md`.
- **Experiment worktrees: share `data/` and `.venv` (MUST):** every
  `git worktree` for an experiment (`attribute-based-person-retrieval-core-<id>`) **must
  not copy** `data/` or `.venv`. Link both to the main checkout
  `attribute-based-person-retrieval-core/` — **no second
  `scripts/download-dataset.py` run, no duplicated dataset trees, no
  second venv install**. Default YAML `data.root: data` is relative to
  that worktree, which is why naive downloads fill the disk. Parent-workspace
  `_KEEP_data_duplicates_DO_NOT_DELETE/` is temporary recovery after a
  Windows junction wipe, not the desired policy; do not create new
  duplicates.

  Immediately after `git worktree add`, before any train/download (Windows:
  prefer junction/`mklink /J`; Git Bash/WSL: symlink):

  ```sh
  # from the new worktree root; MAIN is the primary core checkout
  MAIN="../attribute-based-person-retrieval-core"
  # Git Bash / WSL:
  ln -sfn "$(cd "$MAIN" && pwd)/data" data
  ln -sfn "$(cd "$MAIN" && pwd)/.venv" .venv
  # Windows CMD (admin not required for /J):
  # mklink /J data ..\attribute-based-person-retrieval-core\data
  # mklink /J .venv ..\attribute-based-person-retrieval-core\.venv
  ```

  Or set an absolute data override (also honored by `load_config` and
  `scripts/download-dataset.py`; may live in `.env` — see `.env.example`):

  ```sh
  export WACV_DATA_ROOT="$(cd ../attribute-based-person-retrieval-core && pwd)/data"
  ```

  If neither a local `data/` corpus nor `WACV_DATA_ROOT` is set,
  `resolve_data_root` / `download-dataset.py` automatically fall back to
  sibling `../attribute-based-person-retrieval-core/data` when that tree
  already looks complete — so a fresh worktree must not re-download.
  Still create the `data` and `.venv` junctions/symlinks so tools that
  assume `./data` / `./.venv` keep working. `download-dataset.py` skips
  only when annotations **and** all three corpora (Market1501, PA100k,
  PETA) exist under the resolved root. Never delete or re-download the
  main worktree's `data/`.

  **Before `git worktree remove` (MUST):**
  1. **Keep models and zips in the main checkout first.** Move or copy
     scored or keep-worthy checkpoints (`best.pt` / `last.pt` and any
     package under `build/*.zip`) from this worktree into the main
     checkout at
     `attribute-based-person-retrieval-core/build/_KEEP_<id-or-run>/`
     (example already on disk:
     `build/_KEEP_969857_e101_teacher_identity/`). Do not leave the only
     copy of a Codabench-scored or final-upload candidate inside a
     disposable worktree. Parent-workspace lesson: the local **969857**
     zip and E40 teacher checkpoint were deleted with their worktrees and
     are no longer recoverable from disk.
  2. In that worktree only, remove the `data` and `.venv` reparse points
     (`rmdir data .venv` on Windows CMD for junctions, or `unlink`/`rm`
     the symlinks — never `rm -rf` through them). Then run
     `git worktree remove`. Leaving the junctions in place lets worktree
     cleanup follow them and wipe the main checkout's real `data/` /
     `.venv` (the incident that forced the quarantine recovery).
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
  `submission-rank-smoke.py`), matching the CLI-entrypoint convention
  already established there. This does not apply to `src/wacv/**` -
  Python package/module names there must stay valid identifiers
  (underscores), and cannot use hyphens at all.
- **`src/wacv/` folder layout by concern (MUST):** a concern gets its own
  subpackage once it is more than "one obvious file" - `src/wacv/models/`
  (network architectures; `_common.py` holds the shared head, freeze and
  layer-wise-LR helpers), `src/wacv/evaluation/` (`predict.py` TTA and
  gallery prediction, `calibration.py` temperatures, `distances.py` the
  scorer registry and every ranking distance, `rerank.py` dual softmax,
  `pipeline.py` the one predict -> calibrate -> rank -> score function;
  `__init__.py` only re-exports), `src/wacv/monitoring/` (`metrics.py`
  scoring functions including `mean_accuracy`, `tracking.py` `RunTracker`
  persistence/logging), `src/wacv/remote/` (hosted-GPU training backends). A
  single-file module that does not yet belong to one of these concerns
  (`data.py`, `data_root.py`, `config.py`, `normalization.py`,
  `checkpoint.py`, `losses.py`, `attributes.py`, `query_adapter.py`,
  `visualize.py`, `engine.py`) stays flat rather than being forced into a
  one-file package - do not create a subpackage without a second file that
  actually belongs in it.
- **One source per constant (MUST):** a Literal, default or magic value used
  in two places is defined once and imported. `config.py` owns
  `DistanceName`/`CalibrationMode`/`TTAView`/`DomainName` (and derives
  `KNOWN_DOMAINS` from the Literal), `DEFAULT_IMAGE_SIZE`,
  `MODEL_NATIVE_IMAGE_SIZE` (the fixed crop of each ViT, which
  `ExperimentConfig` checks against `data.image_size`) and
  `DEFAULT_CONFIG_PATH`; `normalization.py` owns the ImageNet and CLIP
  mean/std. `submission/run.py` is the one deliberate exception (it ships
  alone) and reads those values from `model_card.json` instead.
- **Remote code and weights are pinned and safe (MUST):** every
  `torch.hub.load` names an immutable commit (see `DINOV2_HUB_COMMIT`), never
  a branch; every `torch.load` uses `weights_only=True`. Moving to a new
  upstream commit means re-loading a real checkpoint with `strict=True`
  against it before changing the pin.
- **A behavior-preserving change starts from golden outputs (MUST):** a
  refactor, a move or a typing pass is "preserving" only if you can show it.
  Before the first edit, capture outputs from the unchanged code on organizer
  val data and a seeded smoke run: the scorers on seeded arrays, real cached
  probabilities through the full S* pipeline, a CPU `configs/smoke.yml` run
  (per-epoch metrics and `best.pt` tensors), `evaluate.py` with calibration,
  `package-submission.py` then `run.py` `rank_gallery`, and model structure
  (state-dict keys, freeze flags, layer-wise LR groups). Run that harness twice
  on the unchanged code first and confirm the two captures are identical (the
  absolute checkpoint path in `model_card.json` is the one run-specific
  field); then compare bit-for-bit after every commit. The harness is
  verification, not a test suite: keep it in a scratch directory, not in this
  repo. The unit tests stayed green through changes that the goldens would
  have caught, so do not treat a green `pytest` as proof of "unchanged".
- **Read the diff after every scripted or regex edit (MUST):** a
  `sed`/Python rename across files cannot tell the library's name from the
  submission's deliberately different mirror of it. Renaming
  `apply_per_attribute_temperatures` also rewrote
  `submission/run.py`'s private `_apply_per_attribute_temperatures` inside a
  test; only reading `git diff` caught it. Match the exact token, never a
  substring, and read the result.
- **Fix type errors in code, with these patterns (MUST):** no suppression
  comment is ever the answer (see above). What worked in past type-error
  cleanups: `module.get_submodule("a.b")`, `get_parameter` and
  `register_parameter` for dynamic `nn.Module` attributes (they carry a real
  runtime check); `isinstance` narrowing at the call site instead of a `bool`
  helper; a runtime-checkable `Protocol` instead of
  `getattr(x, "name", None)`; a `Protocol` plus `cast` for duck-typed
  backbones; a class-level annotation for a registered buffer; `cast` once,
  naming the real type, when a library stub is wrong (`torch.jit.trace`,
  `torch.hub.load`'s `trust_repo`); a `sys.platform` branch for POSIX-only
  modules; `importlib.import_module` for a module that only exists on a
  runtime path. Type a batch by what it can be (train batches may carry query
  ids, eval batches never do) so `len(batch)` narrows.
- **The edit hook rejects suppression-directive text anywhere (MUST):** also
  inside a doc, a docstring, an emitted-code string or a scratch script. Say
  "a type-checker suppression comment" in prose. Do not emit one into generated
  notebook cells either; the exporter used to.
- **Verification runs on CPU while another run owns the GPU (MUST):** export
  `CUDA_VISIBLE_DEVICES=` (empty) for tests, smoke training, evaluation and
  packaging. Training on the only GPU belongs to the experiment lane; a smoke
  run that competes with it can OOM or slow a multi-hour run.
- **Experiment training resume and checkpoints (MUST):** every experiment
  train config must set `run.resume: true` (resolves to
  `runs/<run.name>/checkpoints/last.pt`). `scripts/train.py` already writes
  `last.pt` and `best.pt` each epoch via `CheckpointManager`; without
  `resume: true`, a killed or interrupted run cannot continue, and a second
  launch into an existing `runs/<name>/` raises `FileExistsError`. A first
  start with no `last.pt` yet is fine (logs a warning and starts at epoch
  0). Do not set `resume: false` or omit resume on multi-hour GPU
  experiment trains. Bounded smoke configs (`configs/smoke.yml`) may leave
  resume off.
- **No new scripts without an explicit ask (MUST):** agents **MUST NOT**
  create new scripts (under `scripts/`, one-off `*.py`/`*.sh` helpers,
  scratch runners anywhere in this repo or its worktrees) unless the
  human **explicitly** asked to create that script in the current
  request. Prefer reusing existing scripts, CLI entrypoints
  (`uv run python scripts/<existing>.py …`), or documented commands.
  Updating an existing script the human named is OK when the task
  requires it; creating a *new* file is not.
- **No unrequested code (MUST):** only add files and logic the current task
  acceptance criteria require. Prefer extending an existing script or
  library module over new packages, fixtures, or README sections. Delete
  scaffolding that is not needed for train / evaluate / package / submit.
  Creating a *new* script still requires an explicit human ask (see the
  bullet above).
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
  `src/`, `scripts/`, `tests/`, `submission/`, `configs/`, and
  `deployments/` file for the exact name before removing it, and note in the
  PR description (or commit body) that the search came back empty.
  `query_adapter.py::unwrap_backbone`, `attach_hard_query_adapter` and the
  whole `wise_ft` cluster were removed this way - a
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
- **`submission/run.py` mirrors the library's dtype and memory shape, not
  just its logic (MUST):** `submission/` is intentionally self-contained
  (no `src.wacv` import - it ships standalone into the sandboxed Codabench
  eval image, see the file's own docstring), so every prediction/ranking
  function it reimplements has to copy the library's numeric behavior
  exactly, not re-derive it. A submission has already failed this way: the
  `run.py` copy of a ranking function kept a full-size float64 matrix after
  the library copy had moved to chunked float32, and Codabench OOM-killed it
  with an opaque platform message, not a Python traceback, while `pytest`
  and the small-gallery Docker smoke both passed (details:
  parent workspace `records/ROUTER.md`, "Submission incidents").
  After touching any submission-side reimplementation of library math,
  run (or add, if the function is new) a parity test like
  `tests/test_dual_softmax.py`'s
  `test_submission_dual_softmax_matches_library` - it dynamically loads
  `submission/run.py` and asserts its output matches the library function's,
  which is what actually catches a
  re-derivation that drifted. A parity test is necessary but not
  sufficient on its own: it proves the two agree on a small array, not
  that either one's *memory* scales to a real gallery size - test at (or
  above) the real gallery shape when the change is about a memory
  footprint rather than a formula.
- **A memory fix for a ranking function is not done until the whole
  ranking stage is profiled at real gallery scale, not just the one
  function that was touched (MUST):** fixing only the suspected function
  once left the real peak elsewhere in the ranking stage, and the next
  upload failed with the same opaque error (parent workspace
  `records/ROUTER.md`, "Submission incidents"). After any change meant to fix or bound a
  submission-side OOM, profile peak RSS across the full call it sits
  inside (`rank_gallery`, not just the one function), at the real
  gallery/query counts recorded in this submission's own
  `records/submissions/*/SUMMARY.md` - a unit test on a small array
  proves the formula, never the memory footprint.
- **Prefer numpy's in-place ops (`+=`, `out=`, mutate-and-return) over the
  naive expression form for any O(Q×G) array on the submission's ranking
  path (MUST):** `a + b - 2*c`, `np.where(cond, x, np.zeros_like(y))`, and
  `weight * a + (1 - weight) * b` each read as a single line but allocate
  one full-size temporary per operator, and on a real gallery several
  hundred-MB temporaries stack up across one expression. `_hybrid_distance`, `_minmax_per_query`, `_nll_distance`,
  and `_l1_distance` (and their identical library copies in
  `src/wacv/evaluation/__init__.py`) held up to four such arrays alive at
  once before this was rewritten to mutate a single scratch buffer
  through each step - which cut measured peak memory on the real shape
  substantially with zero output change (verified against this repo's own
  naive-loop parity tests, not assumed; numbers in the parent workspace's
  `records/ROUTER.md`, "Submission incidents"). Apply this only where
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
  ...`, never a hand-written one-off. `<backend>` is `colab` (try
  first — `google-colab-cli`, **Linux/macOS only**, run from WSL2 on
  Windows; `colab exec` blocks the local process for the whole run, so use
  `tmux`/`nohup` for one that must outlive a closed terminal) or `kaggle`
  (the fallback — cross-platform including native Windows; a pushed
  kernel runs server-side, so it survives a closed laptop, which also
  makes it the better choice for a run known in advance to outlive a
  session). See the next rule for exactly when an agent moves from one to
  the other. Add a third hosted provider by dropping a module under
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

- **Training order is Colab, then Kaggle, then local — and a code bug
  never falls through to the next one (MUST):** start every real training
  run with `remote-training.py colab`. Move to the next option only on an
  **infrastructure failure** — the `colab`/`kaggle` CLI wrapper itself
  exiting non-zero before or during the run: auth (`gcloud`/ADC), quota,
  session/kernel provisioning, network, or a timeout with no result yet.
  On that class of failure, retry with `remote-training.py kaggle`; if
  Kaggle also fails the same way, only then fall back to training on this
  machine (`uv run python scripts/train.py --config configs/<name>.yml`).
  Never start on the local machine by default — both hosted options are
  free GPU time this machine does not have to spend, and skipping
  straight to local silently defeats the reason `remote-training.py`
  exists.

  A **code-logic failure** — a traceback or exception from inside the
  staged training code itself, visible in the downloaded run's logs, a
  crashing import, a NaN loss, a shape mismatch, anything that is a bug in
  *this repo's* code rather than in reaching a GPU — does **not** trigger
  a fallback to the next backend or to local. Colab and Kaggle run the
  identical staged code (see `stage.py` above), so Kaggle would reproduce
  the exact same crash: retrying it burns a second remote quota slot for
  no new information, and a false "it eventually ran" reads as success
  once local training happens to route around the same bug on a
  differently-shaped dev box. Fix the code, verify with the
  notebook/remote-path check above, then restart from Colab.

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

  **Hugging Face Hub (gated models, e.g. EmbeddingGemma):** put a
  read token in `.env` as `HF_TOKEN` (see `.env.example`;
  `HUGGING_FACE_HUB_TOKEN` is an accepted alias). Prefer that over
  interactive `hf auth login` / `huggingface-cli login`, which remains an
  optional fallback when neither env var is set.
  `scripts/download-hf-model.py` loads `.env` the same way
  `remote-training.py` does. Accept the model licence on the Hub page
  before downloading; never commit tokens.

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
  affect runtime behavior, not just imports. The two `--stage-only` runs
  write copies of the source under `build/remote-kaggle` and
  `build/remote-colab`, and pyright scans `build/`: delete both after the
  check (they are disposable), or a stale copy reports errors that are not in
  the tree.

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
- Commit Kaggle/Colab/Hugging Face credentials (`.env`, `kaggle.json`,
  `access_token`, `HF_TOKEN`, OAuth config) anywhere in this repo, or
  `git clone` this repo from inside a Kaggle kernel or Colab VM (stage a
  self-contained folder instead — `src/wacv/remote/stage.py`).

</delivery_gates>

<required>

## Rules learned from past mistakes (MUST)

Rules only. The evidence behind each one (which run, which numbers, when)
lives in the parent workspace's experiment docs and `records/`; look there,
do not copy it here.

- **Check a library claim against the installed source before writing it**
  in code, a docstring or a doc. Read the module in `.venv`; do not rely on
  memory of how a library behaves.
- **Never build a tensor inside a traced `forward`.** `torch.jit.trace`
  freezes it to the tracing device. Use a registered buffer.
- **Smoke every package on the CUDA path, not only on CPU.** `run.py` falls
  back to CPU when its CUDA probe fails, which turns a device bug into a
  timeout. Confirm the smoke log reports CUDA, and compare the package's
  probabilities with the eager model on a few val images.
- **Rebuild or refresh the zip after any `submission/run.py` change,** and
  record its checksum where the package is documented.
- **No performance setting without a measurement** recorded in the
  experiment's docs. Eval loaders do not keep persistent workers.
- **A timing is a proxy only on an idle GPU.** Check `nvidia-smi` first;
  record a timing taken next to another job as contended, never as the proxy.
  Proxy or conservative A10 estimates must not hard-block packaging or
  submit advice by themselves; record them, and hard-block on time only
  with hard Codabench evidence of an official-limit timeout kill. Parent
  workspace counterexample: 3-view E103 (970882) was marked do-not-submit
  on proxy timing yet scored exit 0 on Codabench (see parent
  `records/ROUTER.md`, "Evaluator and package gate"). Whether a candidate
  is **worth packaging / worth a slot** (local vs B*, final bar = beat B*
  test mADM) is decided in the parent `AGENTS.md` ("Worth packaging /
  worth a slot") and `records/ROUTER.md` — do not invent a second gate here.
- **"No behaviour change" needs proof.** Keep the old implementation as a
  test oracle and compare outputs on real data.
- **Assert in the test body,** not inside a helper, so a reader can see what
  fails.
- **After `git commit`, `git status` must be clean of the files you meant to
  include.** Push branches as `git push origin <branch>`; a bare `git push`
  can target `main`.
- **Multi-line edits go in a script file.** Never combine a heredoc with a
  second stdin redirect, and never chain heredocs with `||`.
- **Name by what the caller reads:** plain verbs that match sibling methods,
  `can_`/`is_` for yes/no answers, and the words the ledgers already use.

</required>
