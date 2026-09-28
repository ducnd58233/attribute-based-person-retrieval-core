"""Kaggle GPU-kernel backend for scripts/remote-training.py.

Persistence is the kernel running server-side on Kaggle once pushed,
independent of this process or this machine staying up; polling here is
only for reporting/download timing, not for keeping the kernel alive.

Never clones this private repo from inside the kernel (would need a
checked-in token) - stages a self-contained folder instead (see
src/wacv/remote/stage.py).

Kaggle *script* kernels only materialize ``code_file`` under ``/kaggle/src``
at runtime (sibling folders like ``scripts/`` are not present). The push
path therefore embeds the staged tree as a base64 zip inside
``run_remote.py``, extracts it to a writable root, then runs ``run_inner.py``.
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Final

from src.wacv.remote.stage import ROOT, build_staging_dir

STAGING_DIR: Final[Path] = ROOT / "build" / "remote-kaggle"
OUTPUT_DIR: Final[Path] = ROOT / "build" / "remote-kaggle-output"
_TERMINAL_OK: Final[tuple[str, ...]] = ("complete",)
_TERMINAL_FAIL: Final[tuple[str, ...]] = ("error", "cancel")
_SKIP_BUNDLE_NAMES: Final[frozenset[str]] = frozenset(
    {"run_remote.py", "kernel-metadata.json"}
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


def _cli_env() -> dict[str, str]:
    """UTF-8 env for the Kaggle CLI (avoids Windows charmap failures on logs)."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


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


def _zip_staging_payload(staging_dir: Path) -> bytes:
    """Zip staged pipeline files (everything except the outer entrypoint/metadata)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staging_dir.rglob("*")):
            if not path.is_file() or path.name in _SKIP_BUNDLE_NAMES:
                continue
            archive.write(path, path.relative_to(staging_dir).as_posix())
    payload = buffer.getvalue()
    if not payload:
        raise SystemExit(f"refusing empty Kaggle payload from {staging_dir}")
    return payload


def embed_staging_payload(staging_dir: Path) -> None:
    """Rewrite ``run_remote.py`` so one uploaded file carries the whole tree.

    Kaggle script kernels only expose ``code_file`` under ``/kaggle/src``;
    nested staged folders are not available at runtime. Embed them as a
    base64 zip, extract to a writable root, then execute ``run_inner.py``.
    """
    entry = staging_dir / "run_remote.py"
    if not entry.is_file():
        raise SystemExit(f"missing staged entrypoint: {entry}")

    inner = staging_dir / "run_inner.py"
    inner.write_text(entry.read_text(encoding="utf-8"), encoding="utf-8")
    payload_b64 = base64.b64encode(_zip_staging_payload(staging_dir)).decode("ascii")
    wrapper = _self_extracting_entrypoint(payload_b64=payload_b64)
    compile(wrapper, str(entry), "exec")
    entry.write_text(wrapper, encoding="utf-8")
    # Outer code_file is the wrapper only; inner lives inside the zip.
    inner.unlink(missing_ok=True)


def _self_extracting_entrypoint(*, payload_b64: str) -> str:
    """Single-file kernel source: extract zip, then run ``run_inner.py``."""
    return (
        '"""Auto-generated Kaggle single-file entrypoint - do not hand-edit.\n\n'
        "Kaggle script kernels only materialize this file under /kaggle/src.\n"
        "The staged scripts/src/configs tree is embedded as PAYLOAD_B64 and\n"
        "extracted before run_inner.py (the normal remote stages) runs.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        "import base64\n"
        "import io\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "import zipfile\n"
        "from pathlib import Path\n\n"
        f"PAYLOAD_B64 = {payload_b64!r}\n\n\n"
        "def _extract_root() -> Path:\n"
        "    working = Path('/kaggle/working')\n"
        "    if working.is_dir() and os.access(working, os.W_OK):\n"
        "        return working\n"
        "    return Path.cwd()\n\n\n"
        "def main() -> None:\n"
        "    root = _extract_root()\n"
        "    os.chdir(root)\n"
        "    blob = base64.b64decode(PAYLOAD_B64)\n"
        "    with zipfile.ZipFile(io.BytesIO(blob)) as archive:\n"
        "        archive.extractall(root)\n"
        "    marker = root / 'scripts' / 'download-dataset.py'\n"
        "    if not marker.is_file():\n"
        "        raise FileNotFoundError(f'payload extract failed: missing {marker}')\n"
        "    inner = root / 'run_inner.py'\n"
        "    if not inner.is_file():\n"
        "        raise FileNotFoundError(f'payload extract failed: missing {inner}')\n"
        "    subprocess.run([sys.executable, str(inner)], check=True, cwd=root)\n\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


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
            encoding="utf-8",
            errors="replace",
            env=_cli_env(),
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"transient status check failure, retrying: {exc}")
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
            print(f"[{elapsed_min:6.1f}m] {status_text}")
            terminal_words = _TERMINAL_OK + _TERMINAL_FAIL
            if any(word in status_text.lower() for word in terminal_words):
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
    print(f"staged {result.staging_dir} (run={result.run_name})")
    embed_staging_payload(result.staging_dir)
    print("embedded staged tree into run_remote.py (Kaggle single-file script)")

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
        env=_cli_env(),
    )
    print(f"pushed {kernel_id}; polling every {args.poll_interval:.0f}s")

    final_status = wait_for_terminal_status(
        kernel_id, poll_interval=args.poll_interval, timeout_hours=args.timeout_hours
    )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    output_result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(OUTPUT_DIR), "-o"],
        check=False,
        env=_cli_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if output_result.returncode != 0:
        if output_result.stdout:
            print(output_result.stdout)
        if output_result.stderr:
            print(output_result.stderr)
        print(f"kernel page for logs: https://www.kaggle.com/code/{kernel_id}")
        raise SystemExit(
            f"kaggle kernels output failed (exit {output_result.returncode}); "
            f"kernel status was: {final_status}"
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
