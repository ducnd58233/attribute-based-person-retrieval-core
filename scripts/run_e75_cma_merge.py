"""E75 / H78: CMA-ES over layer-group merge coefficients (organizer val only).

Pre-registered (2026-10-03):
  members = E17, E72(=E25 retrain), E34, E52  (E26 checkpoint absent)
  fitness during search = ID-val mADM, identity TTA, S* (hybrid+dual_softmax)
  final report = ID-val + V-deg mean over L1/L2/L3 (identity TTA)
  popsize=8, maxiter=12, sigma0=0.4, seed=42
  promote if merged beats best member by >= +0.005 on ID-val AND V-deg mean
  never use Codabench / test scores (Terms section 3)

Example:

  uv run python scripts/run_e75_cma_merge.py \\
    --config configs/e75_cma_merge.yml \\
    --out-dir runs/e75_cma_merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.wacv.layer_merge import (  # noqa: E402
    LAYER_GROUPS,
    LayerMergeResult,
    cma_ask,
    cma_tell,
    decode_cma_vector,
    extract_model_state,
    merge_by_layer_groups,
    uniform_logits,
)

LOGGER = logging.getLogger("e75")

DEFAULT_MEMBERS: tuple[tuple[str, Path], ...] = (
    (
        "e17",
        ROOT.parent
        / "attribute-based-person-retrieval-core"
        / "runs"
        / "e17_clip_b16_lpft"
        / "checkpoints"
        / "best.pt",
    ),
    (
        "e72_e25",
        ROOT.parent
        / "attribute-based-person-retrieval-core-e72"
        / "runs"
        / "e72_e25_retrain"
        / "checkpoints"
        / "best.pt",
    ),
    (
        "e34",
        ROOT.parent
        / "attribute-based-person-retrieval-core-e34"
        / "runs"
        / "e34_clip_all"
        / "checkpoints"
        / "best.pt",
    ),
    (
        "e52",
        ROOT.parent
        / "attribute-based-person-retrieval-core-e52"
        / "runs"
        / "e52_label_mask"
        / "checkpoints"
        / "best.pt",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "e75_cma_merge.yml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs" / "e75_cma_merge",
    )
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--sigma0", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stop-gens",
        type=int,
        default=3,
        help="stop if best ID-val mADM improves by <0.001 for this many gens",
    )
    parser.add_argument(
        "--skip-vdeg",
        action="store_true",
        help="skip final V-deg L1-L3 (debug only)",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=5000,
        help="gallery/query cap during CMA fitness (0 = full val)",
    )
    return parser.parse_args()


def _config_with_limit(config_path: Path, limit: int | None, out_path: Path) -> Path:
    """Write a YAML sibling that overrides data.limit for search fitness."""
    text = config_path.read_text(encoding="utf-8")
    if limit is None or limit <= 0:
        out_path.write_text(text, encoding="utf-8")
        return out_path
    # Replace the data.limit line; keep the rest of the fitness config intact.
    lines: list[str] = []
    replaced = False
    for line in text.splitlines():
        if line.strip().startswith("limit:") and not replaced:
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f"{indent}limit: {int(limit)}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        raise ValueError(f"no data.limit key in {config_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _eval_state_dict(
    *,
    config_path: Path,
    state_dict: dict[str, torch.Tensor],
    checkpoint_path: Path,
    degrade: str,
) -> float:
    """Persist weights then score ID-val mADM (optionally under V-deg)."""
    from scripts.evaluate import evaluate_checkpoint

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state_dict}, checkpoint_path)
    results, _config = evaluate_checkpoint(
        config_path,
        checkpoint_path,
        protocols=("id_val",),
        distance=None,
        degrade=degrade,
    )
    key = "id_val" if degrade == "none" else f"id_val_deg_{degrade}"
    return float(results[key]["madm"])


def _member_baselines(
    *,
    config_path: Path,
    members: Sequence[tuple[str, Path]],
    out_dir: Path,
) -> dict[str, float]:
    from scripts.evaluate import evaluate_checkpoint

    scores: dict[str, float] = {}
    for name, path in members:
        LOGGER.info("baseline member %s path=%s", name, path)
        results, _ = evaluate_checkpoint(
            config_path,
            path,
            protocols=("id_val",),
            degrade="none",
        )
        scores[name] = float(results["id_val"]["madm"])
        LOGGER.info("baseline %s ID-val mADM=%.4f", name, scores[name])
        _write_json(out_dir / f"baseline_{name}.json", results)
    return scores


def _write_manifest(
    out_dir: Path,
    *,
    members: Sequence[tuple[str, Path]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "hypothesis": "H78",
        "experiment": "E75",
        "note": "E26 checkpoint absent; four CLIP B/16 members only",
        "code_sha": _code_sha(),
        "members": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in members
        },
        "groups": list(LAYER_GROUPS),
        "population": args.population,
        "maxiter": args.maxiter,
        "sigma0": args.sigma0,
        "seed": args.seed,
        "fitness": "id_val_madm_identity_tta_sstar",
        "search_limit": int(args.search_limit),
        "promotion_bar": 0.005,
        "e25_via_e72": True,
        "e26_missing": True,
    }
    _write_json(out_dir / "config.json", manifest)
    (out_dir / "code.sha").write_text(manifest["code_sha"] + "\n", encoding="utf-8")
    return manifest


@dataclass(frozen=True)
class _SearchCtx:
    config_path: Path
    states: Sequence[Mapping[str, torch.Tensor]]
    names: Sequence[str]
    ckpt_dir: Path
    out_dir: Path
    population: int
    maxiter: int
    sigma0: float
    seed: int
    early_stop_gens: int
    uniform: LayerMergeResult
    uniform_madm: float


def _score_population(
    ctx: _SearchCtx,
    samples: Sequence[Sequence[float]],
    *,
    generation: int,
    best_madm: float,
    best_vector: list[float],
    best_coeffs: tuple[tuple[float, ...], ...],
) -> tuple[
    list[float],
    list[dict[str, Any]],
    float,
    list[float],
    tuple[tuple[float, ...], ...],
]:
    n_groups = len(LAYER_GROUPS)
    n_members = len(ctx.names)
    fitnesses: list[float] = []
    gen_rows: list[dict[str, Any]] = []
    for sample_index, vector in enumerate(samples):
        logits = decode_cma_vector(vector, n_groups=n_groups, n_members=n_members)
        merged = merge_by_layer_groups(ctx.states, ctx.names, logits)
        madm = _eval_state_dict(
            config_path=ctx.config_path,
            state_dict=merged.state_dict,
            checkpoint_path=ctx.ckpt_dir / "candidate.pt",
            degrade="none",
        )
        fitnesses.append(-madm)
        gen_rows.append(
            {
                "sample": sample_index,
                "madm": madm,
                "coefficients": merged.coefficients,
            }
        )
        LOGGER.info("gen=%d sample=%d mADM=%.4f", generation, sample_index, madm)
        if madm > best_madm + 1e-6:
            best_madm = madm
            best_vector = list(vector)
            best_coeffs = merged.coefficients
            torch.save({"model": merged.state_dict}, ctx.ckpt_dir / "best_merged.pt")
    return fitnesses, gen_rows, best_madm, best_vector, best_coeffs


def _run_cma_search(
    ctx: _SearchCtx,
) -> tuple[float, list[float], tuple[tuple[float, ...], ...], list[dict[str, Any]]]:
    dim = len(LAYER_GROUPS) * len(ctx.names)
    history: list[dict[str, Any]] = []
    mean = [0.0] * dim
    cov_diag = [1.0] * dim
    sigma = float(ctx.sigma0)
    rng = torch.Generator()
    rng.manual_seed(int(ctx.seed))

    # Archive is search-limit fitness only (not full-val member baselines).
    best_madm = float(ctx.uniform_madm)
    best_vector = list(mean)
    best_coeffs = ctx.uniform.coefficients
    torch.save({"model": ctx.uniform.state_dict}, ctx.ckpt_dir / "best_merged.pt")
    stagnant = 0

    for generation in range(ctx.maxiter):
        archive_before = best_madm
        samples = cma_ask(
            mean,
            sigma,
            cov_diag,
            population=ctx.population,
            rng=rng,
        )
        fitnesses, gen_rows, best_madm, best_vector, best_coeffs = _score_population(
            ctx,
            samples,
            generation=generation,
            best_madm=best_madm,
            best_vector=best_vector,
            best_coeffs=best_coeffs,
        )
        mean, sigma, cov_diag = cma_tell(mean, sigma, cov_diag, samples, fitnesses)
        gen_best = max(row["madm"] for row in gen_rows)
        stagnant = 0 if best_madm >= archive_before + 0.001 else stagnant + 1
        history.append(
            {
                "generation": generation,
                "sigma": sigma,
                "gen_best_madm": gen_best,
                "archive_best_madm": best_madm,
                "samples": gen_rows,
            }
        )
        _write_json(ctx.out_dir / "history.json", history)
        LOGGER.info(
            "gen=%d done gen_best=%.4f archive_best=%.4f sigma=%.3f stagnant=%d",
            generation,
            gen_best,
            best_madm,
            sigma,
            stagnant,
        )
        if stagnant >= ctx.early_stop_gens:
            LOGGER.info("early stop after %d stagnant generations", stagnant)
            break

    return best_madm, best_vector, best_coeffs, history


def _vdeg_pair(
    *,
    config_path: Path,
    merged_path: Path,
    member_path: Path,
) -> tuple[dict[str, float], dict[str, float], float, float]:
    from scripts.evaluate import evaluate_checkpoint

    merged_vdeg: dict[str, float] = {}
    member_vdeg: dict[str, float] = {}
    for level in ("L1", "L2", "L3"):
        merged_vdeg[level] = _eval_state_dict(
            config_path=config_path,
            state_dict=extract_model_state(str(merged_path)),
            checkpoint_path=merged_path,
            degrade=level,
        )
        LOGGER.info("V-deg %s merged mADM=%.4f", level, merged_vdeg[level])
        results, _ = evaluate_checkpoint(
            config_path,
            member_path,
            protocols=("id_val",),
            degrade=level,
        )
        key = f"id_val_deg_{level}"
        member_vdeg[level] = float(results[key]["madm"])
        LOGGER.info("V-deg %s member mADM=%.4f", level, member_vdeg[level])
    merged_mean = sum(merged_vdeg.values()) / len(merged_vdeg)
    member_mean = sum(member_vdeg.values()) / len(member_vdeg)
    return merged_vdeg, member_vdeg, merged_mean, member_mean


def _build_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble metrics.json from a flat result mapping."""
    best_madm = float(payload["best_madm"])
    best_member_madm = float(payload["best_member_madm"])
    vdeg_mean = float(payload["vdeg_mean"])
    member_vdeg_mean = float(payload["member_vdeg_mean"])
    return {
        "baselines_id_val": dict(payload["baselines"]),
        "best_member": payload["best_member_name"],
        "best_member_id_val": best_member_madm,
        "uniform_soup_id_val": float(payload["uniform_madm"]),
        "uniform_soup_search_madm": float(payload["uniform_search_madm"]),
        "merged_search_madm": float(payload["merged_search_madm"]),
        "merged_id_val": best_madm,
        "merged_coefficients": payload["best_coeffs"],
        "merged_vector": list(payload["best_vector"]),
        "vdeg_merged": dict(payload["vdeg"]),
        "vdeg_merged_mean": vdeg_mean,
        "vdeg_best_member": dict(payload["member_vdeg"]),
        "vdeg_best_member_mean": member_vdeg_mean,
        "delta_id_val": best_madm - best_member_madm,
        "delta_vdeg_mean": vdeg_mean - member_vdeg_mean,
        "promotion_bar": 0.005,
        "promoted": bool(payload["promoted"]),
        "wall_seconds": float(payload["wall_seconds"]),
        "search_limit": int(payload["search_limit"]),
        "e26_missing": True,
        "e25_via_e72": True,
    }


def main() -> None:  # noqa: PLR0915
    from src.wacv.utils import configure_default_logging

    configure_default_logging()
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    members = [(name, path.resolve()) for name, path in DEFAULT_MEMBERS]
    missing = [name for name, path in members if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing member checkpoints: {missing}")

    _write_manifest(out_dir, members=members, args=args)
    full_config = args.config.resolve()
    search_config = _config_with_limit(
        full_config,
        args.search_limit if args.search_limit > 0 else None,
        out_dir / "e75_search_limit.yml",
    )
    states = [extract_model_state(str(path)) for _, path in members]
    names = [name for name, _ in members]

    LOGGER.info("scoring member baselines (full ID-val)")
    baselines = _member_baselines(
        config_path=full_config, members=members, out_dir=out_dir
    )
    best_member_name = max(baselines, key=baselines.get)
    best_member_madm = baselines[best_member_name]
    uniform = merge_by_layer_groups(
        states, names, uniform_logits(len(LAYER_GROUPS), len(names))
    )
    torch.save({"model": uniform.state_dict}, ckpt_dir / "uniform_soup.pt")
    LOGGER.info(
        "scoring uniform soup on search-limit=%s",
        args.search_limit,
    )
    uniform_search_madm = _eval_state_dict(
        config_path=search_config,
        state_dict=uniform.state_dict,
        checkpoint_path=ckpt_dir / "uniform_soup.pt",
        degrade="none",
    )
    LOGGER.info(
        "best member %s=%.4f uniform_search=%.4f",
        best_member_name,
        best_member_madm,
        uniform_search_madm,
    )

    started = time.perf_counter()
    ctx = _SearchCtx(
        config_path=search_config,
        states=states,
        names=names,
        ckpt_dir=ckpt_dir,
        out_dir=out_dir,
        population=args.population,
        maxiter=args.maxiter,
        sigma0=args.sigma0,
        seed=args.seed,
        early_stop_gens=args.early_stop_gens,
        uniform=uniform,
        uniform_madm=uniform_search_madm,
    )
    merged_search_madm, best_vector, best_coeffs, _history = _run_cma_search(ctx)

    LOGGER.info("re-scoring best merged + uniform on full ID-val")
    best_madm = _eval_state_dict(
        config_path=full_config,
        state_dict=extract_model_state(str(ckpt_dir / "best_merged.pt")),
        checkpoint_path=ckpt_dir / "best_merged.pt",
        degrade="none",
    )
    uniform_madm = _eval_state_dict(
        config_path=full_config,
        state_dict=extract_model_state(str(ckpt_dir / "uniform_soup.pt")),
        checkpoint_path=ckpt_dir / "uniform_soup.pt",
        degrade="none",
    )
    LOGGER.info(
        "full ID-val merged=%.4f uniform=%.4f best_member=%.4f",
        best_madm,
        uniform_madm,
        best_member_madm,
    )

    if args.skip_vdeg:
        vdeg: dict[str, float] = {}
        member_vdeg: dict[str, float] = {}
        vdeg_mean = float("nan")
        member_vdeg_mean = float("nan")
    else:
        vdeg, member_vdeg, vdeg_mean, member_vdeg_mean = _vdeg_pair(
            config_path=full_config,
            merged_path=ckpt_dir / "best_merged.pt",
            member_path=dict(members)[best_member_name],
        )

    id_delta = best_madm - best_member_madm
    vdeg_delta = vdeg_mean - member_vdeg_mean
    promoted = (not args.skip_vdeg) and id_delta >= 0.005 and vdeg_delta >= 0.005
    summary = _build_summary(
        {
            "baselines": baselines,
            "best_member_name": best_member_name,
            "best_member_madm": best_member_madm,
            "uniform_madm": uniform_madm,
            "uniform_search_madm": uniform_search_madm,
            "merged_search_madm": merged_search_madm,
            "best_madm": best_madm,
            "best_coeffs": best_coeffs,
            "best_vector": best_vector,
            "vdeg": vdeg,
            "member_vdeg": member_vdeg,
            "vdeg_mean": vdeg_mean,
            "member_vdeg_mean": member_vdeg_mean,
            "promoted": promoted,
            "wall_seconds": time.perf_counter() - started,
            "search_limit": int(args.search_limit),
        }
    )
    _write_json(out_dir / "metrics.json", summary)
    LOGGER.info("E75 summary %s", json.dumps(summary, sort_keys=True))
    print("E75_DONE", json.dumps({"promoted": promoted, "merged_id_val": best_madm}))


if __name__ == "__main__":
    main()
