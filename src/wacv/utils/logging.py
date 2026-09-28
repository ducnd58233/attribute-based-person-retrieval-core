"""Shared root-logger setup for CLI entrypoints.

Call :func:`configure_logging` once at process start. Library modules keep
``logging.getLogger(__name__)``. :class:`~src.wacv.monitoring.tracking.RunTracker`
owns its own non-propagating logger and ``run.log`` handlers; do not route
those through this root config.
"""

from __future__ import annotations

import io
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Console and optional file logging options for the root logger."""

    console_level: int = logging.INFO
    console_enabled: bool = True
    console_logger_levels: Mapping[str, int] = field(default_factory=dict)
    file_path: Path | None = None
    file_level: int = logging.INFO
    file_logger_levels: Mapping[str, int] = field(default_factory=dict)
    fmt: str = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


class _LoggerLevelFilter(logging.Filter):
    """Raise or lower the effective threshold for named logger prefixes."""

    def __init__(self, levels: Mapping[str, int], default_level: int) -> None:
        super().__init__()
        self._levels = dict(levels)
        self._default = default_level

    def filter(self, record: logging.LogRecord) -> bool:
        threshold = self._default
        for prefix, level in self._levels.items():
            if record.name == prefix or record.name.startswith(f"{prefix}."):
                threshold = level
                break
        return record.levelno >= threshold


def configure_logging(config: LoggingConfig) -> None:
    """Reset root handlers and apply console and optional file logging."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt=config.fmt, datefmt=config.datefmt)

    if config.console_enabled:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(config.console_level)
        console.addFilter(
            _LoggerLevelFilter(config.console_logger_levels, config.console_level)
        )
        console.setFormatter(formatter)
        root.addHandler(console)

    if config.file_path is not None:
        config.file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.file_path, encoding="utf-8")
        file_handler.setLevel(config.file_level)
        file_handler.addFilter(
            _LoggerLevelFilter(config.file_logger_levels, config.file_level)
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def configure_stdout_utf8() -> None:
    """Prefer UTF-8 stdout so multilingual generation prints on Windows consoles."""
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def configure_default_logging() -> None:
    """UTF-8 stdout plus INFO-level console logging on the root logger."""
    configure_stdout_utf8()
    configure_logging(LoggingConfig())
