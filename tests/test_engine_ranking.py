"""Per-epoch ranking metrics surface on the Trainer epoch tqdm postfix."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.wacv.config import load_config
from src.wacv.engine import RankingEvalData, Trainer


class _TinyModel(nn.Module):
    def __init__(self, num_attributes: int = 40) -> None:
        super().__init__()
        self.fc = nn.Linear(4, num_attributes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat = images.reshape(images.shape[0], -1)[:, :4]
        return self.fc(flat)


def _tiny_loader(batch_size: int = 2) -> DataLoader:
    images = torch.randn(4, 3, 8, 4)
    targets = torch.randint(0, 2, (4, 40), dtype=torch.float32)
    return DataLoader(TensorDataset(images, targets), batch_size=batch_size)


def test_fit_logs_ranking_and_sets_postfix(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "smoke.yml", project_root=root)
    config = config.model_copy(
        update={
            "optimization": config.optimization.model_copy(update={"epochs": 1}),
            "run": config.run.model_copy(
                update={
                    "name": "rank-epoch-test",
                    "output_dir": tmp_path / "runs",
                    "progress": True,
                    "device": "cpu",
                    "resume": None,
                }
            ),
        }
    )
    train_loader = _tiny_loader()
    val_loader = _tiny_loader()
    gallery_loader = _tiny_loader()
    queries = np.random.randint(0, 2, size=(2, 40), dtype=np.int64)
    gallery_attrs = np.random.randint(0, 2, size=(4, 40), dtype=np.int64)
    ranking = RankingEvalData(
        queries=queries,
        gallery_attrs=gallery_attrs,
        gallery_loader=gallery_loader,
    )

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer)
    criterion = nn.BCEWithLogitsLoss()
    tracker = MagicMock()
    tracker.checkpoint_dir = tmp_path / "ckpt"
    tracker.checkpoint_dir.mkdir()
    tracker.logger = MagicMock()

    trainer = Trainer(
        model, optimizer, scheduler, criterion, torch.device("cpu"), tracker, config
    )
    result = trainer.fit(train_loader, val_loader, ranking=ranking)
    assert result.end_epoch == 0
    logged = tracker.log_metrics.call_args[0][1]
    assert "val/mAP" in logged
    assert "val/Rank-1" in logged
    assert "val/loss" in logged
