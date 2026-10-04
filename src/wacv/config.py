"""Typed experiment configuration loaded from root-level YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self, get_args

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveFloat,
    model_validator,
)
from pydantic.types import NonNegativeInt, PositiveInt

from src.wacv.data_root import resolve_data_root

ModelName = Literal[
    "resnet50",
    "convnext_tiny",
    "convnext_base",
    "clip_vit_b16",
    "clip_vit_l14",
    "dinov2_vitb14_reg",
]
DeviceName = Literal["auto", "cpu", "cuda", "tpu"]
DistanceName = Literal[
    "l1", "specker", "clear_hard", "nll", "hybrid", "group_categorical"
]
GroupMode = Literal["independent", "categorical"]
CalibrationMode = Literal["none", "per_attribute"]
QueryNorm = Literal["none", "dual_softmax"]
TTAView = Literal["identity", "hflip", "scale_crop"]
DomainName = Literal["PA100k", "PETA", "Market1501"]
KNOWN_DOMAINS: tuple[str, ...] = get_args(DomainName)
EvalProtocol = Literal["id_val", "lodo_m"]


def _path_from_yaml(value: object) -> object:
    return Path(value) if isinstance(value, str) else value


ConfigPath = Annotated[Path, BeforeValidator(_path_from_yaml)]


def _pair_from_yaml(value: object) -> object:
    """Accept YAML length-2 lists as pairs."""
    if isinstance(value, list) and len(value) == 2:
        return (value[0], value[1])
    return value


ImageSize = Annotated[tuple[PositiveInt, PositiveInt], BeforeValidator(_pair_from_yaml)]
FloatPair = Annotated[tuple[float, float], BeforeValidator(_pair_from_yaml)]
JpegQualityRange = Annotated[
    tuple[PositiveInt, PositiveInt], BeforeValidator(_pair_from_yaml)
]

# Default (H, W) crop, and the native crop of each fixed-geometry ViT tower.
# The tower builders and the data transforms must agree on it.
DEFAULT_IMAGE_SIZE: tuple[int, int] = (256, 128)
MODEL_NATIVE_IMAGE_SIZE: dict[str, tuple[int, int]] = {
    "clip_vit_b16": (256, 128),
    "clip_vit_l14": (224, 112),
    "dinov2_vitb14_reg": (252, 126),
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "convnext_baseline.yml"


def _resume_from_yaml(value: object) -> object:
    """Accept null/false (off), true (auto last.pt), or a path string."""
    if value is True:
        return True
    if value is False or value is None:
        return None
    if isinstance(value, str):
        return Path(value)
    return value


# True means "resume from runs/<name>/checkpoints/last.pt" after load_config.
ResumeSetting = Annotated[
    Path | Literal[True] | None, BeforeValidator(_resume_from_yaml)
]


class StrictConfig(BaseModel):
    """Forbid silent typos and type coercion in experiment files."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelConfig(StrictConfig):
    name: ModelName = "convnext_base"
    pretrained: bool = True
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    hard_query_adapter: bool = False
    adapter_dim: PositiveInt = 256
    freeze_backbone: bool = False
    lora_rank: NonNegativeInt = Field(
        default=0,
        description="If >0, inject LoRA on CLIP attention out_proj (PEFT).",
    )
    lora_alpha: PositiveFloat = 16.0
    weights_path: ConfigPath | None = Field(
        default=None,
        description=(
            "Optional local backbone checkpoint (e.g. vendored DINOv2 .pth). "
            "When set, skip hub weight download and load this file instead."
        ),
    )


class DegradeAugConfig(StrictConfig):
    """CCTV-style train-time degradations (no hue jitter; colour is labelled)."""

    enabled: bool = False
    downscale: FloatPair = (0.3, 1.0)
    blur_p: float = Field(default=0.3, ge=0.0, le=1.0)
    jpeg: JpegQualityRange = (30, 90)

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        scale_lo, scale_hi = self.downscale
        if not (0.0 < scale_lo <= scale_hi <= 1.0):
            raise ValueError(
                "degrade_aug.downscale must satisfy 0 < min <= max <= 1, "
                f"got {self.downscale}"
            )
        jpeg_lo, jpeg_hi = self.jpeg
        if jpeg_lo > jpeg_hi:
            raise ValueError(f"degrade_aug.jpeg min must be <= max, got {self.jpeg}")
        return self


class DataConfig(StrictConfig):
    root: ConfigPath = Path("data")
    batch_size: PositiveInt = 32
    num_workers: NonNegativeInt = 4
    limit: PositiveInt | None = None
    train_augmentation: bool = True
    train_domains: list[DomainName] | None = None
    eval_domains: list[DomainName] | None = None
    image_size: ImageSize = DEFAULT_IMAGE_SIZE
    degrade_aug: DegradeAugConfig = Field(default_factory=DegradeAugConfig)


class AsmrAuxConfig(StrictConfig):
    """ASMR ICCV 2021 semantic-margin auxiliary loss (train-only)."""

    weight: float = Field(default=0.0, ge=0.0, le=5.0)


class DistillTeacherConfig(StrictConfig):
    """Frozen teacher tower used only during soft+hard KD training."""

    checkpoint: ConfigPath
    name: ModelName = "clip_vit_l14"
    pretrained: bool = False
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    freeze_backbone: bool = True
    lora_rank: NonNegativeInt = 16
    lora_alpha: PositiveFloat = 16.0
    image_size: ImageSize | None = None


class DistillConfig(StrictConfig):
    """Soft+hard knowledge distillation into the student classifier."""

    soft_weight: float = Field(default=0.0, ge=0.0, le=5.0)
    hard_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    temperature: PositiveFloat = 2.0
    teacher: DistillTeacherConfig | None = None

    @model_validator(mode="after")
    def _teacher_required_when_soft(self) -> Self:
        if self.soft_weight > 0.0 and self.teacher is None:
            raise ValueError(
                "losses.distill.soft_weight > 0 requires losses.distill.teacher"
            )
        return self


class LossesConfig(StrictConfig):
    asmr_aux: AsmrAuxConfig = Field(default_factory=AsmrAuxConfig)
    distill: DistillConfig = Field(default_factory=DistillConfig)


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
    early_stop_patience: PositiveInt | None = None
    focal_gamma: float = Field(default=0.0, ge=0.0, le=5.0)
    query_match_weight: float = Field(default=0.0, ge=0.0, le=5.0)
    query_match_temperature: PositiveFloat = 0.07
    lp_ft: bool = Field(
        default=False,
        description="Two-stage linear-probe then full fine-tune schedule.",
    )
    lp_epochs: PositiveInt = 2
    lp_learning_rate: PositiveFloat = 1e-3
    layer_decay: float = Field(
        default=0.65,
        gt=0.0,
        le=1.0,
        description="Layer-wise LR decay for ViT FT (deeper layers keep higher LR).",
    )
    logit_adjust: bool = Field(
        default=False,
        description="Train-time logit offset from priors; eval uses raw logits.",
    )


class RunConfig(StrictConfig):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    output_dir: ConfigPath = Path("runs")
    resume: ResumeSetting = None
    init_weights: ConfigPath | None = None
    seed: int = 42
    device: DeviceName = "auto"
    progress: bool = True


class InferenceConfig(StrictConfig):
    """Test-time prediction options (no training side effects)."""

    tta: list[TTAView] = Field(
        default_factory=lambda: ["identity"],
        description="Views to average probabilities over: identity, hflip, "
        "scale_crop (1.15x bilinear upsample then center-crop).",
    )
    tta_scale: PositiveFloat = Field(
        default=1.15,
        description="Upsample factor for the scale_crop TTA view.",
    )


class RankingConfig(StrictConfig):
    """How gallery attribute probabilities become query distances."""

    distance: DistanceName = "l1"
    temperature: PositiveFloat = 1.5
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight on NLL term when distance=hybrid (rest is L1).",
    )
    group_mode: GroupMode = Field(
        default="independent",
        description="When categorical, exclusive groups use categorical NLL "
        "(multi-hot falls back to Bernoulli).",
    )
    calibration: CalibrationMode = Field(
        default="none",
        description="per_attribute fits one temperature per bit on ID-val "
        "(or in-domain val for LODO-M) and stores assets/calibration.json.",
    )
    query_norm: QueryNorm = Field(
        default="none",
        description="dual_softmax: S' = S + log_softmax over queries per gallery "
        "(no labels; InfoNCE-style hubness reduction).",
    )
    query_norm_tau: PositiveFloat = Field(
        default=0.05,
        description="Temperature for dual_softmax column softmax.",
    )
    clear_fusion: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight on CLEAR cosine term when distance=clear_hard "
        "(rest is Specker L1 on calibrated probs).",
    )


class ExperimentConfig(StrictConfig):
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    losses: LossesConfig = Field(default_factory=LossesConfig)
    ranking: RankingConfig = RankingConfig()
    inference: InferenceConfig = InferenceConfig()
    run: RunConfig

    @model_validator(mode="after")
    def _asmr_requires_hard_query_adapter(self) -> ExperimentConfig:
        if self.losses.asmr_aux.weight > 0.0 and not self.model.hard_query_adapter:
            raise ValueError(
                "losses.asmr_aux.weight > 0 requires model.hard_query_adapter=true"
            )
        return self

    @model_validator(mode="after")
    def _image_size_matches_vit_geometry(self) -> ExperimentConfig:
        native = MODEL_NATIVE_IMAGE_SIZE.get(self.model.name)
        if native is not None and self.data.image_size != native:
            raise ValueError(
                f"model.name={self.model.name} needs data.image_size={native}, "
                f"got {self.data.image_size}"
            )
        return self


def _resolve(path: Path | None, project_root: Path) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (project_root / path).resolve()


def _resolve_resume(
    resume: Path | Literal[True] | None,
    output_dir: Path,
    run_name: str,
    root: Path,
) -> Path | None:
    if resume is True:
        return (output_dir / run_name / "checkpoints" / "last.pt").resolve()
    return _resolve(resume, root)


def load_config(path: Path, *, project_root: Path | None = None) -> ExperimentConfig:
    """Load, validate, and resolve project-relative paths from a YAML file."""
    config_path = path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(payload)
    root = project_root.resolve() if project_root else PROJECT_ROOT

    data = config.data.model_copy(
        update={"root": resolve_data_root(config.data.root, root)}
    )
    model = config.model.model_copy(
        update={"weights_path": _resolve(config.model.weights_path, root)}
    )
    distill = config.losses.distill
    if distill.teacher is not None:
        teacher = distill.teacher.model_copy(
            update={"checkpoint": _resolve(distill.teacher.checkpoint, root)}
        )
        if teacher.checkpoint is None:
            raise ValueError("losses.distill.teacher.checkpoint resolved to None")
        native = MODEL_NATIVE_IMAGE_SIZE.get(teacher.name)
        image_size = teacher.image_size if teacher.image_size is not None else native
        if image_size is None:
            raise ValueError(
                f"losses.distill.teacher.name={teacher.name} needs image_size"
            )
        if native is not None and image_size != native:
            raise ValueError(
                f"losses.distill.teacher.name={teacher.name} needs "
                f"image_size={native}, got {image_size}"
            )
        teacher = teacher.model_copy(update={"image_size": image_size})
        distill = distill.model_copy(update={"teacher": teacher})
    losses = config.losses.model_copy(update={"distill": distill})
    output_dir = _resolve(config.run.output_dir, root)
    if output_dir is None:
        raise ValueError("run.output_dir resolved to None")
    run = config.run.model_copy(
        update={
            "output_dir": output_dir,
            "resume": _resolve_resume(
                config.run.resume, output_dir, config.run.name, root
            ),
            "init_weights": _resolve(config.run.init_weights, root),
        }
    )
    return config.model_copy(
        update={"model": model, "data": data, "run": run, "losses": losses}
    )
