"""Loads remote-training credentials from a .env file.

Thin wrapper around python-dotenv rather than a hand-rolled parser: it is
official, pure Python (no compiled extension, so it installs the same way on
Windows and Linux via uv), and already gets the "real env var wins" rule
right through ``override=False``.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv as _load_dotenv


def load_dotenv(path: Path) -> None:
    """Set KEY=VALUE lines from ``path`` into os.environ, real env wins."""
    _load_dotenv(path, override=False)
