"""Unit tests for shared root logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.wacv.utils import LoggingConfig, configure_logging
from src.wacv.utils.logging import _LoggerLevelFilter


def test_logger_level_filter_uses_prefix_override() -> None:
    filt = _LoggerLevelFilter({"src.wacv.remote": logging.WARNING}, logging.INFO)
    info = logging.LogRecord(
        name="src.wacv.remote.kaggle",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="quiet",
        args=(),
        exc_info=None,
    )
    warning = logging.LogRecord(
        name="src.wacv.remote.kaggle",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="loud",
        args=(),
        exc_info=None,
    )
    other = logging.LogRecord(
        name="scripts.train",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="ok",
        args=(),
        exc_info=None,
    )
    assert filt.filter(info) is False
    assert filt.filter(warning) is True
    assert filt.filter(other) is True


def test_configure_logging_writes_file_and_respects_console_filter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "logs" / "app.log"
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    try:
        configure_logging(
            LoggingConfig(
                console_level=logging.INFO,
                console_logger_levels={"noisy": logging.ERROR},
                file_path=log_path,
                file_level=logging.DEBUG,
            )
        )
        root_child = logging.getLogger("scripts.train")
        noisy = logging.getLogger("noisy.child")
        root_child.info("train-info")
        root_child.debug("train-debug")
        noisy.info("should-not-console")
        noisy.error("should-console")

        captured = capsys.readouterr().out
        assert "train-info" in captured
        assert "train-debug" not in captured
        assert "should-not-console" not in captured
        assert "should-console" in captured

        text = log_path.read_text(encoding="utf-8")
        assert "train-info" in text
        assert "train-debug" in text
        assert "should-not-console" in text
        assert "should-console" in text
    finally:
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)


def test_logging_config_defaults_use_ascii_hyphen() -> None:
    assert "—" not in LoggingConfig().fmt
    assert " - " in LoggingConfig().fmt


def test_configure_logging_does_not_touch_run_tracker_logger() -> None:
    """RunTracker uses a non-propagating logger with its own run.log handlers."""
    run_logger = logging.getLogger("wacv.run.test-isolation")
    run_logger.handlers.clear()
    run_logger.propagate = False
    sentinel = logging.NullHandler()
    run_logger.addHandler(sentinel)
    try:
        configure_logging(LoggingConfig(console_enabled=True))
        assert run_logger.handlers == [sentinel]
        assert run_logger.propagate is False
    finally:
        run_logger.removeHandler(sentinel)
        logging.getLogger().handlers.clear()
