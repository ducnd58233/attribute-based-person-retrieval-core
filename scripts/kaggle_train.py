"""Push a training run to a Kaggle GPU kernel, wait for it, pull it back.

Usage:
  uv run python scripts/kaggle_train.py --config configs/smoke.yml --stage-only
  uv run python scripts/kaggle_train.py --config configs/convnext_baseline.yml

``--stage-only`` builds the staging folder (which AST-parses every reachable
stage script and ``src/wacv`` module) and exits, needing no ``kaggle`` CLI or
credentials — this is what AGENTS.md's notebook/remote-path check rule runs.
Without it, this pushes a real Kaggle kernel: persistence is the kernel
running server-side on Kaggle once pushed, independent of this process or
this machine staying up; polling here is only for reporting/download timing.

See AGENTS.md "Remote GPU training (Kaggle, Colab)" for the full walkthrough,
including where credentials come from and what this never does (never clones
this private repo from inside the kernel; never commits credentials).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Final

from remote_stage import ROOT, build_staging_dir

STAGING_DIR: Final[Path] = ROOT / "build" / "kaggle-kernel"
OUTPUT_DIR: Final[Path] = ROOT / "build" / "kaggle-kernel-output"
_TERMINAL_OK: Final[tuple[str, ...]] = ("complete",)
_TERMINAL_FAIL: Final[tuple[str, ...]] = ("error", "cancel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--kaggle-username", default=None)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=None,
        help="Give up polling after this long (the kernel itself keeps "
        "running on Kaggle; re-run `kaggle kernels status`/`output` by hand "
        "later). Default: poll until the kernel reaches a terminal state.",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Build the staging folder and exit; no kaggle CLI or credentials needed.",
    )
    return parser.parse_args()


def resolve_username(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_user = os.environ.get("KAGGLE_USERNAME")
    if env_user:
        return env_user
    creds_path = Path.home() / ".kaggle" / "kaggle.json"
    if creds_path.is_file():
        payload = json.loads(creds_path.read_text(encoding="utf-8"))
        username = payload.get("username")
        if username:
            return str(username)
    raise SystemExit(
        "no Kaggle username: pass --kaggle-username, set KAGGLE_USERNAME, "
        "or put one in ~/.kaggle/kaggle.json (see AGENTS.md)"
    )


def write_kernel_metadata(staging_dir: Path, *, kernel_id: str, title: str) -> None:
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": "run_remote.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
    }
    (staging_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def kernel_status(kernel_id: str) -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel_id],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def wait_for_terminal_status(
    kernel_id: str, *, poll_interval: float, timeout_hours: float | None
) -> str:
    started = time.monotonic()
    while True:
        status_text = kernel_status(kernel_id)
        lowered = status_text.lower()
        elapsed_min = (time.monotonic() - started) / 60
        print(f"[{elapsed_min:6.1f}m] {status_text}")
        if any(word in lowered for word in _TERMINAL_OK + _TERMINAL_FAIL):
            return status_text
        timed_out = (
            timeout_hours is not None
            and (time.monotonic() - started) > timeout_hours * 3600
        )
        if timed_out:
            print(
                f"local poll timed out after {timeout_hours}h; the kernel keeps "
                f"running on Kaggle. Check later: kaggle kernels status {kernel_id}"
            )
            return status_text
        time.sleep(poll_interval)


def pull_run_into_local(output_dir: Path, run_name: str) -> Path | None:
    for candidate in output_dir.rglob(run_name):
        if candidate.is_dir() and candidate.parent.name == "runs":
            destination = ROOT / "runs" / run_name
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(candidate, destination, dirs_exist_ok=True)
            return destination
    return None


def main() -> None:
    args = parse_args()
    result = build_staging_dir(args.config, STAGING_DIR)
    print(f"staged {result.staging_dir} (run={result.run_name})")

    if args.stage_only:
        return

    if shutil.which("kaggle") is None:
        raise SystemExit(
            "kaggle CLI not on PATH; `uv sync` after adding it to the dev "
            "group, or `uv run python scripts/kaggle_train.py ...`"
        )

    username = resolve_username(args.kaggle_username)
    kernel_slug = result.run_name.replace("_", "-").lower()
    kernel_id = f"{username}/{kernel_slug}"
    write_kernel_metadata(
        result.staging_dir, kernel_id=kernel_id, title=result.run_name
    )

    subprocess.run(
        [
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(result.staging_dir),
            "--accelerator",
            args.accelerator,
        ],
        check=True,
    )
    print(f"pushed {kernel_id}; polling every {args.poll_interval:.0f}s")

    final_status = wait_for_terminal_status(
        kernel_id, poll_interval=args.poll_interval, timeout_hours=args.timeout_hours
    )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(OUTPUT_DIR), "-o"],
        check=True,
    )
    landed = pull_run_into_local(OUTPUT_DIR, result.run_name)

    failed = any(word in final_status.lower() for word in _TERMINAL_FAIL)
    if failed or landed is None:
        print(f"kernel page for logs: https://www.kaggle.com/code/{kernel_id}")
        if failed:
            raise SystemExit(f"kernel did not complete: {final_status}")
        raise SystemExit(
            f"kernel completed but runs/{result.run_name}/ was not in its output"
        )

    print(f"done: {landed}")


if __name__ == "__main__":
    main()
