"""Headless remote-GPU training backends for scripts/remote-training.py.

Each backend module (kaggle, colab, ...) exposes a ``register(subparsers)``
function that adds its own argparse subcommand. Add a new remote (a new
hosted GPU provider) by dropping a module here with that same shape and
listing it in scripts/remote-training.py's BACKENDS.
"""
