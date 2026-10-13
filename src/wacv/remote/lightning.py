"""Lightning AI Studio GPU backend for scripts/remote-training.py.

Lightning's free plan needs no credit card (phone verification only) and
grants monthly credits that cover a few T4 runs. Like ``colab exec``, the run
blocks this process until the remote command returns, so keep the terminal
alive (or start it as a runtime job).

The ``lightning-sdk`` package is not a project dependency: adding it would
resync the shared ``.venv`` that every experiment worktree junctions to. The
SDK runs instead in a throwaway interpreter, ``uv run --no-project --with
lightning-sdk==<pin>``, executing ``lightning_driver.py`` from this package.

Credentials come from ``.env`` / the environment: ``LIGHTNING_USER_ID`` and
``LIGHTNING_API_KEY`` (lightning.ai -> Settings -> Keys), and
``LIGHTNING_USERNAME`` with ``LIGHTNING_TEAMSPACE``, which the SDK reads to
find the teamspace the Studio lives in.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Final

from src.wacv.remote.stage import ROOT, build_staging_dir

STAGING_DIR: Final[Path] = ROOT / "build" / "remote-lightning"
DRIVER: Final[Path] = Path(__file__).resolve().with_name("lightning_driver.py")
SDK_SPEC: Final[str] = "lightning-sdk==2026.10.13"
REQUIRED_ENV: Final[tuple[str, ...]] = (
    "LIGHTNING_USER_ID",
    "LIGHTNING_API_KEY",
    "LIGHTNING_USERNAME",
    "LIGHTNING_TEAMSPACE",
)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "lightning", help="Run on a Lightning AI Studio GPU via lightning-sdk"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--machine", default="T4")
    parser.add_argument("--studio", default="wacv-train")
    parser.add_argument(
        "--interruptible",
        action="store_true",
        help="Use a cheaper spot machine; the run restarts from last.pt if lost.",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Build the staging folder and exit; no SDK or credentials needed.",
    )
    parser.set_defaults(func=run)


def require_credentials() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise SystemExit(
            f"missing {', '.join(missing)}: set them in .env (see .env.example); "
            "the API key comes from lightning.ai -> Settings -> Keys"
        )
    if shutil.which("uv") is None:
        raise SystemExit("uv not on PATH; it runs the isolated lightning-sdk driver")


def run(args: argparse.Namespace) -> None:
    result = build_staging_dir(args.config, STAGING_DIR)
    print(f"staged {result.staging_dir} (run={result.run_name})")

    if args.stage_only:
        return

    require_credentials()
    command = [
        "uv",
        "run",
        "--no-project",
        "--with",
        SDK_SPEC,
        "python",
        str(DRIVER),
        "--staging-dir",
        str(result.staging_dir),
        "--run-name",
        result.run_name,
        "--machine",
        args.machine,
        "--studio",
        args.studio,
        "--runs-dir",
        str(ROOT / "runs"),
    ]
    if args.interruptible:
        command.append("--interruptible")
    subprocess.run(command, check=True)
    print(f"done: {ROOT / 'runs' / result.run_name}")
