"""Google Colab GPU-session backend for scripts/remote-training.py.

Unlike Kaggle's server-side kernels, ``colab exec`` blocks this process in
the foreground for the whole run: closing this terminal (or the machine
going to sleep) stops the training run. For a run that must survive a closed
laptop, run this under `tmux`/`nohup` on the Linux/macOS/WSL2 side, or
prefer the ``kaggle`` backend.

Requires the ``google-colab-cli`` tool (`uv tool install google-colab-cli`),
which supports Linux/macOS only - run from WSL2 on Windows.

``colab upload`` accepts a single *file* (not a directory). ``colab exec -f``
reads a *local* file and sends its source to the remote kernel. Large
self-extracting scripts have dropped the websocket, so this backend uploads a
``payload.zip`` plus a tiny bootstrap, then execs the bootstrap.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Final

from src.wacv.remote.stage import ROOT, build_staging_dir

STAGING_DIR: Final[Path] = ROOT / "build" / "remote-colab"
PAYLOAD_NAME: Final[str] = "payload.zip"
BOOTSTRAP_NAME: Final[str] = "colab_bootstrap.py"

_BOOTSTRAP_SOURCE: Final[str] = """\
\"\"\"Auto-generated Colab bootstrap - do not hand-edit.\"\"\"
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/content")
PAYLOAD = ROOT / "payload.zip"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not PAYLOAD.is_file():
        raise FileNotFoundError(f"missing uploaded payload: {PAYLOAD}")
    with zipfile.ZipFile(PAYLOAD) as archive:
        archive.extractall(ROOT)
    entry = ROOT / "run_remote.py"
    if not entry.is_file():
        raise FileNotFoundError(f"payload extract failed: missing {entry}")
    subprocess.run([sys.executable, str(entry)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
"""


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
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=16.0,
        help="Pass-through to `colab exec --timeout` (seconds = hours * 3600).",
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


def _write_payload_zip(staging_dir: Path) -> Path:
    """Zip the staged tree into one uploadable file under ``staging_dir``."""
    zip_path = staging_dir / PAYLOAD_NAME
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staging_dir.rglob("*")):
            if not path.is_file() or path.name in {PAYLOAD_NAME, BOOTSTRAP_NAME}:
                continue
            archive.write(path, path.relative_to(staging_dir).as_posix())
    if zip_path.stat().st_size == 0:
        raise SystemExit(f"refusing empty Colab payload from {staging_dir}")
    return zip_path


def _write_bootstrap(staging_dir: Path) -> Path:
    bootstrap = staging_dir / BOOTSTRAP_NAME
    bootstrap.write_text(_BOOTSTRAP_SOURCE, encoding="utf-8")
    compile(_BOOTSTRAP_SOURCE, str(bootstrap), "exec")
    return bootstrap


def run(args: argparse.Namespace) -> None:
    result = build_staging_dir(args.config, STAGING_DIR)
    print(f"staged {result.staging_dir} (run={result.run_name})")

    if args.stage_only:
        return

    require_supported_platform()
    session = args.session
    colab = ["colab", "--auth", args.auth]
    payload = _write_payload_zip(result.staging_dir)
    bootstrap = _write_bootstrap(result.staging_dir)
    timeout_s = max(args.timeout_hours, 0.1) * 3600.0

    subprocess.run([*colab, "new", "-s", session, "--gpu", args.gpu], check=True)
    try:
        subprocess.run(
            [*colab, "upload", "-s", session, str(payload), PAYLOAD_NAME],
            check=True,
        )
        subprocess.run(
            [*colab, "upload", "-s", session, str(bootstrap), BOOTSTRAP_NAME],
            check=True,
        )
        subprocess.run(
            [
                *colab,
                "exec",
                "-s",
                session,
                "-f",
                str(bootstrap),
                "--timeout",
                str(timeout_s),
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
                f"runs/{result.run_name}",
                str(local_runs),
            ],
            check=True,
        )
    finally:
        subprocess.run([*colab, "stop", "-s", session], check=False)

    print(f"done: {ROOT / 'runs' / result.run_name}")
