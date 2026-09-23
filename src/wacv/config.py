"""Typed experiment configuration loaded from root-level YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PositiveFloat
from pydantic.types import NonNegativeInt, PositiveInt

ModelName = Literal["resnet50", "convnext_tiny", "convnext_base"]
DeviceName = Literal["auto", "cpu", "cuda"]


def _path_from_yaml(value: object) -> object:
    return Path(value) if isinstance(value, str) else value


ConfigPath = Annotated[Path, BeforeValidator(_path_from_yaml)]


class StrictConfig(BaseModel):
    """Forbid silent typos and type coercion in experiment files."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelConfig(StrictConfig):
    name: ModelName = "convnext_base"
    pretrained: bool = True
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


class DataConfig(StrictConfig):
    root: ConfigPath = Path("data")
    batch_size: PositiveInt = 32
    num_workers: NonNegativeInt = 4
    limit: PositiveInt | None = None
    train_augmentation: bool = True


class OptimizationConfig(StrictConfig):
    epochs: PositiveInt = 20
    learning_rate: PositiveFloat = 1e-4
    weight_decay: float = Field(default=1e-2, ge=0.0)
    pos_weight_clip: PositiveFloat = 10.0
    amp: bool = True
    grad_clip_norm: PositiveFloat | None = 1.0
    ema_decay: float = Field(default=0.999, gt=0.0, lt=1.0)
    scheduler_factor: float = Field(default=0.1, gt=0.0, lt=1.0)
    scheduler_patience: NonNegativeInt = 3
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=0.5)


class RunConfig(StrictConfig):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    output_dir: ConfigPath = Path("runs")
    resume: ConfigPath | None = None
    seed: int = 42
    device: DeviceName = "auto"
    progress: bool = True


class ExperimentConfig(StrictConfig):
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    run: RunConfig


def _resolve(path: Path | None, project_root: Path) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (project_root / path).resolve()


def load_config(path: Path, *, project_root: Path | None = None) -> ExperimentConfig:
    """Load, validate, and resolve project-relative paths from a YAML file."""
    config_path = path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(payload)
    root = (
        project_root.resolve() if project_root else Path(__file__).resolve().parents[2]
    )

    data = config.data.model_copy(update={"root": _resolve(config.data.root, root)})
    run = config.run.model_copy(
        update={
            "output_dir": _resolve(config.run.output_dir, root),
            "resume": _resolve(config.run.resume, root),
        }
    )
    return config.model_copy(update={"data": data, "run": run})
