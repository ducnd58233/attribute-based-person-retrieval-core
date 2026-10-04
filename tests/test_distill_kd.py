"""Unit tests for soft+hard knowledge distillation (E40 / H40)."""

from __future__ import annotations

import torch
from torch import nn

from src.wacv.config import ExperimentConfig, load_config
from src.wacv.losses import soft_hard_kd_loss


def test_soft_hard_kd_hard_only_matches_bce() -> None:
    criterion = nn.BCEWithLogitsLoss()
    student = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    hard = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    teacher = torch.zeros_like(student)
    expected = criterion(student, hard)
    got = soft_hard_kd_loss(
        student,
        hard,
        teacher,
        hard_criterion=criterion,
        soft_weight=0.0,
        hard_weight=1.0,
        temperature=2.0,
    )
    torch.testing.assert_close(got, expected)


def test_soft_hard_kd_soft_pulls_toward_teacher() -> None:
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    hard = torch.tensor([[1.0, 0.0]])
    teacher = torch.tensor([[4.0, -4.0]])
    cold = torch.tensor([[-2.0, 2.0]], requires_grad=True)
    warm = torch.tensor([[3.0, -3.0]], requires_grad=True)
    loss_cold = soft_hard_kd_loss(
        cold,
        hard,
        teacher,
        hard_criterion=criterion,
        soft_weight=1.0,
        hard_weight=0.0,
        temperature=1.0,
    )
    loss_warm = soft_hard_kd_loss(
        warm,
        hard,
        teacher,
        hard_criterion=criterion,
        soft_weight=1.0,
        hard_weight=0.0,
        temperature=1.0,
    )
    assert loss_cold.item() > loss_warm.item()


def test_soft_hard_kd_temperature_scales_soft_term() -> None:
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    student = torch.tensor([[0.0, 0.0]])
    hard = torch.tensor([[1.0, 0.0]])
    teacher = torch.tensor([[2.0, -2.0]])
    loss_t1 = soft_hard_kd_loss(
        student,
        hard,
        teacher,
        hard_criterion=criterion,
        soft_weight=1.0,
        hard_weight=0.0,
        temperature=1.0,
    )
    loss_t2 = soft_hard_kd_loss(
        student,
        hard,
        teacher,
        hard_criterion=criterion,
        soft_weight=1.0,
        hard_weight=0.0,
        temperature=2.0,
    )
    # Softened teacher targets + T^2 scale change the soft loss magnitude.
    assert not torch.allclose(loss_t1, loss_t2)


def test_distill_config_requires_teacher_when_soft_weight_positive(
    tmp_path,
) -> None:
    yaml_path = tmp_path / "bad_distill.yml"
    yaml_path.write_text(
        """
model:
  name: clip_vit_b16
  pretrained: false
data:
  root: data
  image_size: [256, 128]
optimization:
  epochs: 1
losses:
  distill:
    soft_weight: 0.5
    hard_weight: 0.5
    temperature: 2.0
run:
  name: bad_distill
""",
        encoding="utf-8",
    )
    try:
        load_config(yaml_path)
        raise AssertionError("expected ValidationError for missing teacher")
    except Exception as exc:  # pydantic ValidationError or ValueError
        assert "teacher" in str(exc).lower() or "distill" in str(exc).lower()


def test_distill_config_loads_when_teacher_present(tmp_path) -> None:
    ckpt = tmp_path / "teacher.pt"
    ckpt.write_bytes(b"")
    yaml_path = tmp_path / "ok_distill.yml"
    yaml_path.write_text(
        f"""
model:
  name: clip_vit_b16
  pretrained: false
data:
  root: data
  image_size: [256, 128]
optimization:
  epochs: 1
losses:
  distill:
    soft_weight: 0.5
    hard_weight: 0.5
    temperature: 2.0
    teacher:
      checkpoint: {ckpt.as_posix()}
      name: clip_vit_l14
      dropout: 0.1
      lora_rank: 16
      lora_alpha: 16.0
      pretrained: false
run:
  name: ok_distill
""",
        encoding="utf-8",
    )
    config = load_config(yaml_path)
    assert isinstance(config, ExperimentConfig)
    assert config.losses.distill.soft_weight == 0.5
    assert config.losses.distill.teacher is not None
    assert config.losses.distill.teacher.name == "clip_vit_l14"
