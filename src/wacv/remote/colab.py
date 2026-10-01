"""Google Colab GPU-session backend for scripts/remote-training.py.

Unlike Kaggle's server-side kernels, ``colab exec`` blocks this process in
the foreground for the whole run: closing this terminal (or the machine
going to sleep) stops the training run. For a run that must survive a closed
laptop, run this under `tmux`/`nohup` on the Linux/macOS/WSL2 side, or
prefer the ``kaggle`` backend.

Requires the ``google-colab-cli`` tool (`uv tool install google-colab-cli`),
which supports Linux/macOS only - run from WSL2 on Windows.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Final

from src.wacv.remote.stage import ROOT, build_staging_dir

STAGING_DIR: Final[Path] = ROOT / "build" / "remote-colab"
REMOTE_DIR: Final[str] = "wacv-remote-train"


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "colab", help="Drive a Google Colab GPU session via google-colab-cli"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--session", default="wacv-train")
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Build the staging folder and exit; no colab CLI or auth needed.",
    )
    parser.add_argument(
        "--auth",
        choices=("adc", "oauth2"),
        default="adc",
        help="Pass-through to `colab --auth` (default adc for unattended runs).",
    )
    parser.set_defaults(func=run)


def require_supported_platform() -> None:
    if platform.system() == "Windows":
        raise SystemExit(
            "google-colab-cli supports Linux/macOS only (not Windows). Run "
            "this from WSL2, or use the `kaggle` backend instead - see "
            "AGENTS.md 'Remote GPU training (Kaggle, Colab)'."
        )
    if shutil.which("colab") is None:
        raise SystemExit(
            "colab CLI not on PATH; install with `uv tool install google-colab-cli`"
        )


def run(args: argparse.Namespace) -> None:
    result = build_staging_dir(args.config, STAGING_DIR)
    print(f"staged {result.staging_dir} (run={result.run_name})")

    if args.stage_only:
        return

    require_supported_platform()
    session = args.session
    colab = ["colab", "--auth", args.auth]

    subprocess.run([*colab, "new", "-s", session, "--gpu", args.gpu], check=True)
    try:
        subprocess.run(
            [
                *colab,
                "upload",
                "-s",
                session,
                str(result.staging_dir),
                REMOTE_DIR,
            ],
            check=True,
        )
        subprocess.run(
            [
                *colab,
                "exec",
                "-s",
                session,
                "-f",
                f"{REMOTE_DIR}/run_remote.py",
            ],
            check=True,
        )
        local_runs = ROOT / "runs" / result.run_name
        # Parent only: `colab download` copies REMOTE's contents *into* LOCAL
        # when LOCAL already exists, which would nest runs/<name>/<name>/...
        # Leaving the final path itself absent makes it become that exact copy.
        (ROOT / "runs").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                *colab,
                "download",
                "-s",
                session,
                f"{REMOTE_DIR}/runs/{result.run_name}",
                str(local_runs),
            ],
            check=True,
        )
    finally:
        subprocess.run([*colab, "stop", "-s", session], check=False)

    print(f"done: {ROOT / 'runs' / result.run_name}")
