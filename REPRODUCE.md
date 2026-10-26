# Reproducing the final entry (WACV 2027 UPAR Challenge, Track 2)

The final entry is Codabench submission **995287**: a probability average of
nine attribute classifiers, ranked by the training-free S\* scorer. Test mADM
**0.664994** (MEVID 0.5242, Synth 0.8058), 750 s inference on the NVIDIA A10.

This page gives two routes:

- **A. Run the exact scored package** (download, check, run).
- **B. Rebuild it from scratch** (train the nine members, package them).

## The package

| Item | Value |
| --- | --- |
| File | `e139_clip_l_dinov2_family_balanced_avg.zip` (5,404,816,330 bytes) |
| Download | <https://drive.google.com/file/d/1mKA3IKlYItip03fe3XCPlhsPRDfAvGB1/view?usp=sharing> |
| SHA-256 | `9518bfd246c97ac50b40fa70d35c52d35a005c72b3892268820b73c53f3d0d6a` |
| Contents | `run.py`, `metadata.yaml`, `assets/model_card.json`, `assets/model_0.pt` (seven CLIP ViT-L/14 members pre-averaged, fp16 TorchScript), `assets/model_1.pt` (DINOv2 ViT-L/14-reg, fp32), `assets/model_2.pt` (DINOv2 ViT-B/14-reg, fp32) |
| Weights in the average | 0.50 for the CLIP group (1/14 per member), 0.25 DINOv2-L, 0.25 DINOv2-B |
| Inference | original view only, bfloat16 autocast on CUDA, 20 decode threads, no gallery smoothing, CPU fallback |

`submission/run.py` in this repository is the `run.py` inside that zip (same
bytes apart from line endings).

## The S\* ranking stage

Each member outputs 40 attribute logits per gallery image; the package averages
the members' sigmoid probabilities `p` with the weights above. For a query
`q` (40 bits) the score has no learned parameters:

1. `D_NLL = -sum_a [q_a log p_a + (1 - q_a) log(1 - p_a)]` and
   `D_L1 = sum_a |q_a - p_a|`.
2. Each is min-max normalised per query over the gallery, then mixed:
   `D = 0.5 * D_NLL + 0.5 * D_L1` (`ranking.distance: hybrid`, `alpha: 0.5`).
3. Dual softmax over the query set: with `S = -D`,
   `S' = S + log_softmax_over_queries(S / 0.05)` (`query_norm: dual_softmax`,
   `query_norm_tau: 0.05`); the gallery is ranked by `-S'` for each query.

Code: `src/wacv/evaluation/distances.py` (`nll_distance`, `l1_distance`,
`hybrid_distance`), `src/wacv/evaluation/rerank.py` (`dual_softmax_rerank`),
and their copies in `submission/run.py`. No re-ranking, neighbour smoothing or
other pass over the gallery is used (`neighbour_smooth: null`). The repository
keeps a kNN neighbour-smoothing experiment (`src/wacv/evaluation/smoothing.py`,
measured on the organizer validation split only); it is off unless a config
sets `ranking.neighbour_smooth`, and no submitted package ever enabled it.

## Setup

```sh
conda env create -f environment.yml   # once; Python 3.13, CUDA 13.0 toolkit
conda activate wacv                   # before every uv command
uv sync
uv run python scripts/download-dataset.py
```

`download-dataset.py` clones the organizers' starter kit
(<https://github.com/speckean/UPAR-Challenge-2027>) for the annotations and
downloads Market1501, PA100k and PETA into `data/`. The training code uses
only the organizers' training split; the hidden test set is never used.

The development environment is newer than the evaluation image. Packages are
run in the official `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` image
through `deployments/docker/`.

Pretrained backbones are fetched on first use and are not trained for
pedestrian attributes:

| Backbone | Source |
| --- | --- |
| CLIP ViT-L/14 (OpenAI) | `open_clip`, architecture `ViT-L-14`, tag `openai` |
| DINOv2 ViT-L/14 and ViT-B/14 with registers | `torch.hub` `facebookresearch/dinov2` at commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, entries `dinov2_vitl14_reg` and `dinov2_vitb14_reg` |

## A. Run the scored package

```sh
sha256sum e139_clip_l_dinov2_family_balanced_avg.zip   # must match the table above
mkdir -p build/e139_clip_l_dinov2_family_balanced_avg
unzip e139_clip_l_dinov2_family_balanced_avg.zip -d build/e139_clip_l_dinov2_family_balanced_avg

# load check, network disabled, official image
SUBMISSION_DIR=../../build/e139_clip_l_dinov2_family_balanced_avg \
  docker compose -f deployments/docker/docker-compose.yml run --rm submission-test

# rank_gallery on the organizer validation gallery (needs data/)
SUBMISSION_DIR=../../build/e139_clip_l_dinov2_family_balanced_avg DATA_DIR=../../data \
  docker compose -f deployments/docker/docker-compose.yml run --rm submission-rank-smoke
```

The zip can also be uploaded to the Codabench Track 2 page as it is.

## B. Rebuild from scratch

### 1. Train the nine members

Each member trains on the PA100k and PETA images of the organizers' training
split (Market1501 is left out on purpose so it can serve as a held-out
domain). After every epoch the EMA weights are scored on the validation
split, and `runs/<run.name>/checkpoints/best.pt` keeps the best epoch. Every
config resumes from `last.pt` when a run is interrupted.

```sh
uv run python scripts/train.py --config configs/<config>.yml
```

| Member | Config | Backbone and adaptation | Training-time change vs. the base pipeline |
| --- | --- | --- | --- |
| E107 | `configs/e107_teacher_degrade.yml` | CLIP ViT-L/14, frozen, LoRA r=16 on attention `out_proj` | downscale-upscale, blur, JPEG |
| E112 | `configs/e112_teacher_clahe.yml` | same | CLAHE on the LAB L channel, before the resize |
| E113 | `configs/e113_teacher_randaugment.yml` | same | RandAugment (N=2, M=9) instead of AugMix |
| E114 | `configs/e114_teacher_random_affine.yml` | same | RandomAffine (p=0.5), before the resize |
| E117 | `configs/e117_teacher_trivialaugment.yml` | same | TrivialAugmentWide instead of AugMix |
| E118 | `configs/e118_teacher_stronger_erasing.yml` | same | Random Erasing p=0.5 (base 0.25) |
| E121 | `configs/e121_teacher_autocontrast_equalize.yml` | same | RandomAutocontrast + RandomEqualize, before the resize |
| E103 | `configs/e103_dinov2_vitl14_lora.yml` | DINOv2 ViT-L/14-reg, frozen, LoRA r=16 on `attn.proj` | none |
| E127 | `configs/e127_dinov2b_partner.yml` | DINOv2 ViT-B/14-reg, LP-FT (2 epochs probe, 8 epochs full, layer decay 0.65) | none |

Base pipeline: resize, AugMix, horizontal flip, ColorJitter 0.2, normalise,
Random Erasing p=0.25. Loss: BCE with positive weights `min((1-p)/p, 10)` and
label smoothing 0.05. Optimiser: AdamW, weight decay 0.01, EMA 0.999, seed 42.

The CLIP members were trained with batch size 8 on one 8 GB GPU; a DINOv2-L
member takes about 8 hours for 8 epochs. GPU training is not bit-for-bit
deterministic, so a retrained member lands close to, not exactly on, the
numbers below. The E103 config was rewritten after its original was lost; the
model it builds loads the shipped E103 weights strictly and gives identical
outputs.

### 2. Check each member locally (optional)

```sh
uv run python scripts/evaluate.py --config configs/<config>.yml --protocol both
```

ID-val uses the official validation queries; LODO-M uses the Market1501 part
of the validation gallery, a domain no member trained on. mADM with the
original view and the S\* scorer:

| Member | ID-val | LODO-M |
| --- | ---: | ---: |
| E107 | 0.7219 | 0.7290 |
| E112 | 0.7226 | 0.7231 |
| E113 | 0.7248 | 0.7234 |
| E114 | 0.7208 | 0.7255 |
| E117 | 0.7229 | 0.7247 |
| E118 | 0.7240 | 0.7220 |
| E121 | 0.7230 | 0.7235 |
| E103 | 0.7285 | 0.7330 |
| E127 | 0.7135 | 0.7173 |
| Final average | 0.7488 | 0.7474 |

`evaluate.py` scores the views a config lists. The E103 config lists three
(original, flip, 1.15x centre crop), which is how its checkpoint was chosen;
with those views E103 scores 0.7302 / 0.7340. The table and the package use
the original view only.

### 3. Package

Three member packages first, then the weighted average of the three. All
packages use the original view only, no gallery smoothing, 20 decode
threads and bfloat16 autocast.

```sh
COMMON="--tta identity --no-neighbour-smooth --decode-workers 20 --amp bf16"

# the seven CLIP members, averaged inside one TorchScript, stored in fp16
uv run python scripts/package-submission.py --config configs/e107_teacher_degrade.yml \
  --checkpoint runs/e107_teacher_degrade/checkpoints/best.pt \
  --member runs/e112_teacher_clahe/checkpoints/best.pt \
  --member runs/e113_teacher_randaugment/checkpoints/best.pt \
  --member runs/e114_teacher_random_affine/checkpoints/best.pt \
  --member runs/e117_teacher_trivialaugment/checkpoints/best.pt \
  --member runs/e118_teacher_stronger_erasing/checkpoints/best.pt \
  --member runs/e121_teacher_autocontrast_equalize/checkpoints/best.pt \
  --weights-dtype fp16 $COMMON --out build/e134_clip_l_seven

uv run python scripts/package-submission.py --config configs/e103_dinov2_vitl14_lora.yml \
  --checkpoint runs/e103_dinov2_vitl14_lora/checkpoints/best.pt $COMMON --out build/e103_member

uv run python scripts/package-submission.py --config configs/e127_dinov2b_partner.yml \
  --checkpoint runs/e127_dinov2b_partner/checkpoints/best.pt $COMMON --out build/e127_member

# the final entry: family-balanced weights 0.50 / 0.25 / 0.25
uv run python scripts/package-submission.py \
  --from-member-dir build/e134_clip_l_seven \
  --from-member-dir build/e103_member \
  --from-member-dir build/e127_member \
  --member-weight 0.5 --member-weight 0.25 --member-weight 0.25 \
  $COMMON --out build/e139_clip_l_dinov2_family_balanced_avg
```

The last command writes `build/e139_clip_l_dinov2_family_balanced_avg.zip`.
Check it with the two Docker commands in route A, then upload it.

## Scored result

| Metric | Value |
| --- | ---: |
| mADM | 0.664994 |
| mAP | 0.384840 |
| Rank-1 | 0.405130 |
| Rank-5 | 0.629985 |
| Rank-10 | 0.690658 |
| mINP | 0.270536 |
| MEVID mADM | 0.5242 |
| Synth mADM | 0.8058 |
| Inference (A10) | 750.12 s |
