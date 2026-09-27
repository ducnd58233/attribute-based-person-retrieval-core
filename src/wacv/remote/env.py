"""Minimal .env loader for remote-training credentials.

Not python-dotenv: the format needed here is two flat KEY=VALUE lines
(KAGGLE_USERNAME, KAGGLE_KEY), not worth a dependency for.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Set KEY=VALUE lines from ``path`` into os.environ, real env wins."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
