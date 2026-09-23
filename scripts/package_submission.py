"""CLI: assemble build/submission/ from template + checkpoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if __name__ == "__main__":
    runpy.run_module("src.wacv.package_submission", run_name="__main__")
