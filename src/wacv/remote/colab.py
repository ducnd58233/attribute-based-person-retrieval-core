"""Google Colab GPU-session backend for scripts/remote-training.py.

Unlike Kaggle's server-side kernels, ``colab exec`` blocks this process in
the foreground for the whole run: closing this terminal (or the machine
going to sleep) stops the training run. For a run that must survive a closed
laptop, run this under `tmux`/`nohup` on the Linux/macOS/WSL2 side, or
prefer the ``kaggle`` backend.

Requires the ``google-colab-cli`` tool (`uv tool install google-colab-cli`),
which supports Linux/macOS only - run from WSL2 on Windows.

``colab upload`` only accepts a single *file*, and a relative remote path
lands at the VM root (``/``), not under ``/content``. This backend therefore
tars the staging folder and uploads to an absolute ``/content/...`` path,
then extracts before ``exec``.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Final

from src.wacv.remote.stage import ROOT, build_staging_dir

STAGING_DIR: Final[Path] = ROOT / "build" / "remote-colab"
REMOTE_DIR_NAME: Final[str] = "wacv-remote-train"
REMOTE_DIR: Final[str] = f"/content/{REMOTE_DIR_NAME}"
REMOTE_TAR: Final[str] = f"/content/{REMOTE_DIR_NAME}.tar.gz"


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


def _tar_staging(staging_dir: Path, tar_path: Path) -> None:
    """Pack ``staging_dir`` so extract yields ``REMOTE_DIR_NAME/`` at /content."""
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(staging_dir, arcname=REMOTE_DIR_NAME)


def _colab(*args: str) -> list[str]:
    """Headless ADC auth; relative remote upload paths land at ``/``, not /content."""
    return ["colab", "--auth", "adc", *args]


def _write_extract_script(path: Path) -> None:
    path.write_text(
        "import tarfile\n"
        "from pathlib import Path\n"
        f"root = Path('/content')\n"
        f"tar = Path({REMOTE_TAR!r})\n"
        "if not tar.is_file():\n"
        "    raise SystemExit(f'missing {tar}')\n"
        "tarfile.open(tar).extractall(root)\n"
        f"out = root / {REMOTE_DIR_NAME!r}\n"
        "print('ok', sorted(p.name for p in out.iterdir()))\n",
        encoding="utf-8",
    )


def _write_train_script(path: Path) -> None:
    path.write_text(
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"cwd = Path({REMOTE_DIR!r})\n"
        "subprocess.run(\n"
        "    [sys.executable, str(cwd / 'run_remote.py')],\n"
        "    check=True,\n"
        "    cwd=str(cwd),\n"
        ")\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    result = build_staging_dir(args.config, STAGING_DIR)
    print(f"staged {result.staging_dir} (run={result.run_name})")

    if args.stage_only:
        return

    require_supported_platform()
    session = args.session

    subprocess.run(_colab("new", "-s", session, "--gpu", args.gpu), check=True)
    try:
        with tempfile.TemporaryDirectory(prefix="wacv-colab-") as tmp:
            tar_path = Path(tmp) / f"{REMOTE_DIR_NAME}.tar.gz"
            extract_py = Path(tmp) / "extract_remote.py"
            train_py = Path(tmp) / "train_remote.py"
            _tar_staging(result.staging_dir, tar_path)
            _write_extract_script(extract_py)
            _write_train_script(train_py)
            subprocess.run(
                _colab("upload", "-s", session, str(tar_path), REMOTE_TAR),
                check=True,
            )
            subprocess.run(
                _colab(
                    "exec",
                    "-s",
                    session,
                    "-f",
                    str(extract_py),
                    "--timeout",
                    "600",
                ),
                check=True,
            )
            # Verify landing before the long train exec.
            subprocess.run(
                _colab("ls", "-s", session, "/content"),
                check=True,
            )
            subprocess.run(
                _colab("exec", "-s", session, "-f", str(train_py)),
                check=True,
            )
        local_runs = ROOT / "runs" / result.run_name
        # Parent only: `colab download` copies REMOTE's contents *into* LOCAL
        # when LOCAL already exists, which would nest runs/<name>/<name>/...
        # Leaving the final path itself absent makes it become that exact copy.
        (ROOT / "runs").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            _colab(
                "download",
                "-s",
                session,
                f"{REMOTE_DIR}/runs/{result.run_name}",
                str(local_runs),
            ),
            check=True,
        )
    finally:
        subprocess.run(_colab("stop", "-s", session), check=False)

    print(f"done: {ROOT / 'runs' / result.run_name}")
