"""Loader and per-epoch ranking speedups must not change what is computed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from src.wacv.config import DataConfig, InferenceConfig, RankingConfig
from src.wacv.data import AttributeDataset, build_loader
from src.wacv.evaluation import (
    RankingEvalData,
    compute_ranking_metrics,
    ranking_metrics_from_probabilities,
    ranking_needs_only_probabilities,
)


def _dataset(tmp_path: Path, count: int = 6) -> AttributeDataset:
    rows = []
    for index in range(count):
        name = f"PA100k/{index}.jpg"
        (tmp_path / "PA100k").mkdir(exist_ok=True)
        Image.new("RGB", (8, 16), color=(index * 30, 0, 0)).save(tmp_path / name)
        rows.append({"image": name, "a": index % 2, "b": 1 - index % 2})
    gt = pd.DataFrame(rows)
    transform = transforms.Compose([transforms.ToTensor()])
    return AttributeDataset(gt, tmp_path, transform)


class _RedModel(nn.Module):
    """Two logits from mean red: deterministic, input-dependent."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        red = images[:, 0].mean(dim=(1, 2))
        return torch.stack([red * 4 - 2, 2 - red * 4], dim=1)


def test_dataset_labels_and_images_follow_gt_order(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    assert dataset.images[2] == "PA100k/2.jpg"
    image, labels = dataset[3]
    assert image.shape == (3, 16, 8)
    assert labels.tolist() == [1.0, 0.0]


def test_eval_loader_uses_eval_batch_size(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    config = DataConfig(batch_size=2, num_workers=0, eval_batch_size=4)
    cpu = torch.device("cpu")
    training_loader = build_loader(dataset, config, cpu, training=True)
    eval_loader = build_loader(dataset, config, cpu, training=False)
    assert training_loader.batch_size == 2
    assert eval_loader.batch_size == 4


def test_scoring_reused_probabilities_matches_full_pass(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    loader = build_loader(
        dataset, DataConfig(num_workers=0), torch.device("cpu"), training=False
    )
    labels = np.array([[i % 2, 1 - i % 2] for i in range(6)], dtype=np.int64)
    data = RankingEvalData(
        queries=np.array([[1, 0], [0, 1]], dtype=np.int64),
        gallery_attrs=labels,
        gallery_loader=loader,
        gallery_images=dataset.images,
    )
    ranking = RankingConfig()
    model = _RedModel()
    full = compute_ranking_metrics(
        model, data, torch.device("cpu"), ranking, progress=False
    )
    with torch.inference_mode():
        probabilities = np.concatenate(
            [torch.sigmoid(model(images)).numpy() for images, _ in loader]
        )
    reused = ranking_metrics_from_probabilities(probabilities, data, ranking)
    assert full.keys() == reused.keys()
    for key, value in full.items():
        assert np.isclose(value, reused[key]), key


def test_reuse_needs_identity_tta_and_no_smoothing() -> None:
    ranking = RankingConfig()
    assert ranking_needs_only_probabilities(ranking, InferenceConfig())
    hflip = InferenceConfig.model_validate({"tta": ["identity", "hflip"]})
    assert not ranking_needs_only_probabilities(ranking, hflip)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_traced_probability_average_runs_on_cuda_after_cpu_trace() -> None:
    from src.wacv.models.ensemble import ProbabilityAverage

    torch.manual_seed(0)
    members = [nn.Linear(4, 3), nn.Linear(4, 3)]
    model = ProbabilityAverage(members, [0.5, 0.5]).eval()
    inputs = torch.randn(5, 4)
    with torch.inference_mode():
        expected = model(inputs)
    # The trace shares parameters with ``model``, so compare to the eager
    # output taken before the dtype and device moves below.
    traced = torch.jit.trace(model, torch.zeros(1, 4))
    traced.half()
    traced.float()
    with torch.inference_mode():
        on_cuda = traced.cuda()(inputs.cuda()).cpu()
    assert torch.allclose(torch.sigmoid(on_cuda), torch.sigmoid(expected), atol=1e-3)
