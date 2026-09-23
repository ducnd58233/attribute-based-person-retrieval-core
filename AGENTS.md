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
  conda activate cv-foundation          # every new shell, before any uv command
  uv sync                               # now safe to run
  ```

  Running `uv` without an active `cv-foundation` conda environment is a
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

</delivery_gates>
