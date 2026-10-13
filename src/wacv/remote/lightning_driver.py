"""Drive one Lightning AI Studio run; executed by ``lightning.py`` only.

This file runs in an isolated ``uv run --with lightning-sdk`` interpreter,
so it imports nothing from this repo and loads the SDK by name: the project
environment does not carry ``lightning-sdk`` (see ``lightning.py``).
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

REMOTE_DIR = "wacv-remote-train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--studio", required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--interruptible", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sdk = importlib.import_module("lightning_sdk")
    machine = getattr(sdk.Machine, args.machine)
    studio = sdk.Studio(name=args.studio, create_ok=True)
    studio.start(machine, interruptible=args.interruptible)
    try:
        studio.upload_folder(str(args.staging_dir), REMOTE_DIR)
        # Studio.run raises RuntimeError with the remote output on a non-zero
        # exit, so a failed stage stops here instead of downloading nothing.
        studio.run(f"cd {REMOTE_DIR} && python run_remote.py")
        args.runs_dir.mkdir(parents=True, exist_ok=True)
        studio.download_folder(
            f"{REMOTE_DIR}/runs/{args.run_name}", str(args.runs_dir / args.run_name)
        )
    finally:
        studio.stop()


if __name__ == "__main__":
    main()
