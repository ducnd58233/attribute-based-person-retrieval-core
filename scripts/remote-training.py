"""Push a training run to a remote GPU backend (Kaggle, Colab, ...).

Usage:
  uv run python scripts/remote-training.py kaggle \
      --config configs/smoke.yml --stage-only
  uv run python scripts/remote-training.py colab \
      --config configs/smoke.yml --gpu T4

``--stage-only`` builds the self-contained staging folder (source + one
experiment YAML, never ``data/``) and exits - no backend CLI or credentials
needed. Without it, this pushes a real remote run; see each backend's own
``--help`` (``remote-training.py kaggle --help``) and AGENTS.md "Remote GPU
training (Kaggle, Colab)" for the full walkthrough.

Credentials load from ``.env`` if present (copy ``.env.example`` to ``.env``,
which is gitignored) into the environment, without overriding a real env var
already set. Never committed, never printed.

Backends live under ``src/wacv/remote/``: each module exposes a
``register(subparsers)`` that adds its own subcommand. Add a new remote GPU
provider by dropping a module there with that same shape and listing it in
BACKENDS below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    from src.wacv.remote import colab, kaggle

    backends = {"kaggle": kaggle, "colab": colab}
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="backend", required=True)
    for module in backends.values():
        module.register(subparsers)
    return parser.parse_args()


def main() -> None:
    from src.wacv.remote.env import load_dotenv

    load_dotenv(ROOT / ".env")
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
