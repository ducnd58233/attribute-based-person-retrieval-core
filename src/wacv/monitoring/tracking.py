"""Small local experiment tracker with MLflow-style run artifacts."""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from src.wacv.config import ExperimentConfig

# Matches ExperimentConfig.model_dump's dotted keys for its Path fields.
_PATH_FIELDS: tuple[str, ...] = ("run.output_dir", "data.root", "run.resume")


def _record_config(config: ExperimentConfig, root: Path) -> dict[str, object]:
    """Dump config for config.json with path fields root-relative when possible.

    Written for portability across environments (e.g. a local dev machine vs
    Colab) that resolve the same YAML to different absolute paths. Nothing
    downstream reads config.json back in, so this only affects the record.
    """
    payload = config.model_dump(mode="json")
    for dotted in _PATH_FIELDS:
        section, field = dotted.split(".")
        value = payload[section].get(field)
        if value is None:
            continue
        absolute = Path(value)
        with suppress(ValueError):  # outside root: keep absolute (e.g. external resume)
            payload[section][field] = absolute.relative_to(root).as_posix()
    return payload


class RunTracker:
    """Persist resolved config, aggregate metrics, and logs in one run folder."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.run_dir = config.run.output_dir / config.run.name
        is_resume = config.run.resume is not None
        if self.run_dir.exists() and not is_resume:
            raise FileExistsError(
                f"Run already exists: {self.run_dir}. Set run.resume to continue it."
            )

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.logger = self._build_logger()

        root = Path(__file__).resolve().parents[3]
        config_path = self.run_dir / "config.json"
        config_path.write_text(
            json.dumps(_record_config(config, root), indent=2) + "\n",
            encoding="utf-8",
        )

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"wacv.run.{self.run_dir.resolve()}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(self.run_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        return logger

    def log_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        record: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "epoch": epoch,
            **metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        summary = ", ".join(f"{key}={value:.6f}" for key, value in metrics.items())
        self.logger.info("epoch=%d | %s", epoch, summary)

    def close(self) -> None:
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    def __enter__(self) -> RunTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()
