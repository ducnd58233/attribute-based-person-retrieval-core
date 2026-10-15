#!/usr/bin/env python3
"""Download a Hugging Face model repo once and reuse it.

    python scripts/download-hf-model.py google/embeddinggemma-300m
    python scripts/download-hf-model.py \\
        https://huggingface.co/google/embeddinggemma-300m
    python scripts/download-hf-model.py org/name --revision <sha> \\
        --include "*.safetensors" "*.json"
    python scripts/download-hf-model.py google/embeddinggemma-300m \\
        --agree-gated-terms

Files land in runs/hf/models/<model-name>, where <model-name> is the repo
name without the org. A finished download writes .download-complete, and
later calls reuse it unless --force is given.

Gated repos (e.g. Gemma): accept the model licence on the Hugging Face
page while logged in, put a read token in ``.env`` as ``HF_TOKEN`` (see
``.env.example``; ``HUGGING_FACE_HUB_TOKEN`` is also accepted), or run
``hf auth login`` / ``huggingface-cli login`` as a fallback, then pass
``--agree-gated-terms`` so this script records an explicit opt-in for the
gated download mechanics. That flag does not accept Google terms for you
on the Hub; it only unblocks the local CLI after you have done the Hub
gate. Never commit tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DEST = REPO_ROOT / "runs" / "hf" / "models"
MARKER = ".download-complete"
AGREE_MARKER = ".gated-terms-agreed"
REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def parse_repo_id(value: str) -> str:
    value = value.strip().rstrip("/")
    value = re.sub(r"^https?://huggingface\.co/", "", value)
    value = re.sub(r"/(tree|blob|resolve)/.*$", "", value)
    if not REPO_ID.match(value):
        raise SystemExit(f"not a Hugging Face repo id or URL: {value!r}")
    return value


def hub_token() -> str | bool:
    """Env token if set; True lets huggingface_hub use a cached ``hf auth login``."""
    return (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or True
    )


def _env_token() -> str | None:
    """Non-empty Hub token from the environment, if any."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


def _auth_hint(*, for_gated_download: bool = False) -> str:
    """Prefer env token; only mention interactive login when none is loaded."""
    if _env_token() is not None:
        if for_gated_download:
            return (
                "HF_TOKEN is set, but this account is not authorized for the "
                "gated repo yet: open the model page on huggingface.co while "
                "logged into the same account as the token and Accept the licence"
            )
        return "HF_TOKEN is already set in the environment / .env"
    return (
        "set HF_TOKEN in .env (copy from .env.example), or export "
        "HF_TOKEN / HUGGING_FACE_HUB_TOKEN; only if neither is set, run "
        "`hf auth login` as a fallback"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo", help="org/name or a huggingface.co URL")
    parser.add_argument("--revision", help="branch, tag or commit sha (default: main)")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"parent dir (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        help="only download files matching these globs",
    )
    parser.add_argument("--exclude", nargs="+", help="skip files matching these globs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again even if already complete",
    )
    parser.add_argument(
        "--agree-gated-terms",
        action="store_true",
        help=(
            "confirm you already accepted this repo's gated licence on "
            "Hugging Face and have Hub auth (HF_TOKEN in .env, or env "
            "export, or hf auth login). Required before downloading a "
            "gated repo."
        ),
    )
    return parser


def _is_gated(repo_id: str) -> bool:
    from huggingface_hub import HfApi

    try:
        info = HfApi(token=hub_token()).model_info(repo_id)
        return bool(getattr(info, "gated", False))
    except Exception:
        # Unknown until snapshot_download; treat as open until proven gated.
        return False


def _write_agree_marker(agree_marker: Path, repo_id: str) -> None:
    agree_marker.write_text(
        json.dumps(
            {
                "repo": repo_id,
                "agreed": True,
                "note": (
                    "Hub gate accepted by operator; " "CLI opt-in for gated download"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_complete_marker(
    marker: Path,
    *,
    repo_id: str,
    revision: str | None,
    agree_gated: bool,
) -> None:
    marker.write_text(
        json.dumps(
            {
                "repo": repo_id,
                "revision": revision or "main",
                "gated_terms_agreed": agree_gated,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    from src.wacv.remote.env import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    args = _build_parser().parse_args()

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError as err:
        raise SystemExit(
            "huggingface_hub is missing: pip install -U huggingface_hub"
        ) from err

    repo_id = parse_repo_id(args.repo)
    target = args.dest / repo_id.split("/", 1)[1]
    marker = target / MARKER
    agree_marker = target / AGREE_MARKER

    if marker.exists() and not args.force:
        print(f"already downloaded: {target}")
        print(target)
        return 0

    target.mkdir(parents=True, exist_ok=True)

    if _is_gated(repo_id) and not args.agree_gated_terms and not agree_marker.exists():
        print(
            f"{repo_id} is gated on Hugging Face. Accept the licence at "
            f"https://huggingface.co/{repo_id}, "
            f"{_auth_hint(for_gated_download=True)}, "
            "then re-run with --agree-gated-terms.",
            file=sys.stderr,
        )
        return 2

    if args.agree_gated_terms:
        _write_agree_marker(agree_marker, repo_id)

    try:
        path = snapshot_download(
            repo_id=repo_id,
            revision=args.revision,
            local_dir=target,
            allow_patterns=args.include,
            ignore_patterns=args.exclude,
            token=hub_token(),
        )
    except GatedRepoError:
        print(
            f"{repo_id} is gated: {_auth_hint(for_gated_download=True)}. "
            f"Model page: https://huggingface.co/{repo_id}. "
            "Then re-run with --agree-gated-terms.",
            file=sys.stderr,
        )
        return 2
    except RepositoryNotFoundError:
        print(
            f"{repo_id} not found, or private and no valid token was given "
            f"({_auth_hint()}).",
            file=sys.stderr,
        )
        return 2

    _write_complete_marker(
        marker,
        repo_id=repo_id,
        revision=args.revision,
        agree_gated=bool(args.agree_gated_terms or agree_marker.exists()),
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
