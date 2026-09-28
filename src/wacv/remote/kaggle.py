"""Kaggle GPU-kernel backend for scripts/remote-training.py.

Persistence is the kernel running server-side on Kaggle once pushed,
independent of this process or this machine staying up; polling here is
only for reporting/download timing, not for keeping the kernel alive.

Never clones this private repo from inside the kernel (would need a
checked-in token) - stages a self-contained folder instead (see
src/wacv/remote/stage.py).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Final

from src.wacv.remote.stage import ROOT, build_staging_dir

logger = logging.getLogger(__name__)

STAGING_DIR: Final[Path] = ROOT / "build" / "remote-kaggle"
OUTPUT_DIR: Final[Path] = ROOT / "build" / "remote-kaggle-output"

# Match Kaggle's KernelWorkerStatus enum only - never bare substring "error",
# which false-positives on ConnectTimeoutError / HTTPSConnectionPool text when
# a poller concatenates CLI stderr into the status string.
_TERMINAL_STATUS_RE: Final[re.Pattern[str]] = re.compile(
    r"KernelWorkerStatus\.(COMPLETE|ERROR|CANCEL\w*)",
    re.IGNORECASE,
)
_FAIL_STATUS_RE: Final[re.Pattern[str]] = re.compile(
    r"KernelWorkerStatus\.(ERROR|CANCEL\w*)",
    re.IGNORECASE,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "kaggle", help="Push to a Kaggle GPU kernel, wait for it, pull it back"
    )
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
    parser.set_defaults(func=run)


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
        "no Kaggle username: pass --kaggle-username, set KAGGLE_USERNAME "
        "(e.g. via .env, see .env.example), or put one in ~/.kaggle/kaggle.json"
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


def is_terminal_status(status_text: str) -> bool:
    """True when status reports a finished KernelWorkerStatus (not infra noise)."""
    return _TERMINAL_STATUS_RE.search(status_text) is not None


def is_failed_status(status_text: str) -> bool:
    """True when status reports KernelWorkerStatus ERROR or CANCEL*."""
    return _FAIL_STATUS_RE.search(status_text) is not None


def kernel_status(kernel_id: str) -> str | None:
    """The kernel's status text, or None on a transient CLI/network hiccup.

    A poll loop that must run for hours cannot let one bad call (a dropped
    connection, a rate limit) take down local tracking of a kernel that is
    still running fine on Kaggle - it retries instead.
    """
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("transient status check failure, retrying: %s", exc)
        return None
    return result.stdout.strip()


def wait_for_terminal_status(
    kernel_id: str, *, poll_interval: float, timeout_hours: float | None
) -> str:
    started = time.monotonic()
    last_status = "unknown (no successful status check yet)"
    while True:
        status_text = kernel_status(kernel_id)
        if status_text is not None:
            last_status = status_text
            elapsed_min = (time.monotonic() - started) / 60
            logger.info("[%6.1fm] %s", elapsed_min, status_text)
            if is_terminal_status(status_text):
                return status_text
        timed_out = (
            timeout_hours is not None
            and (time.monotonic() - started) > timeout_hours * 3600
        )
        if timed_out:
            logger.warning(
                "local poll timed out after %sh; the kernel keeps running on "
                "Kaggle. Check later: kaggle kernels status %s",
                timeout_hours,
                kernel_id,
            )
            return last_status
        time.sleep(poll_interval)


def pull_run_into_local(output_dir: Path, run_name: str) -> Path | None:
    for candidate in output_dir.rglob(glob.escape(run_name)):
        if candidate.is_dir() and candidate.parent.name == "runs":
            destination = ROOT / "runs" / run_name
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(candidate, destination, dirs_exist_ok=True)
            return destination
    return None


def run(args: argparse.Namespace) -> None:
    result = build_staging_dir(args.config, STAGING_DIR)
    logger.info("staged %s (run=%s)", result.staging_dir, result.run_name)

    if args.stage_only:
        return

    if shutil.which("kaggle") is None:
        raise SystemExit(
            "kaggle CLI not on PATH; `uv sync` after adding it to the dev "
            "group, or `uv run python scripts/remote-training.py kaggle ...`"
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
    logger.info("pushed %s; polling every %.0fs", kernel_id, args.poll_interval)

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

    failed = is_failed_status(final_status)
    if failed or landed is None:
        logger.error("kernel page for logs: https://www.kaggle.com/code/%s", kernel_id)
        if failed:
            raise SystemExit(f"kernel did not complete: {final_status}")
        raise SystemExit(
            f"kernel completed but runs/{result.run_name}/ was not in its output"
        )

    logger.info("done: %s", landed)
