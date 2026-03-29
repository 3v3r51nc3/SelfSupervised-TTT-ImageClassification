"""
Base trainer.

Contains shared loop structure:
- train loop,
- validation loop,
- checkpoint/log hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.checkpoint import CheckpointManager
from src.utils.logger import ExperimentLogger


class BaseTrainer(ABC):
    """Shared epoch loop that all stage-specific trainers extend."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        device: torch.device,
        logger: ExperimentLogger,
        checkpoint_mgr: CheckpointManager,
        epochs: int,
        checkpoint_filename: str = "best.pt",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.checkpoint_mgr = checkpoint_mgr
        self.epochs = epochs
        self.checkpoint_filename = checkpoint_filename

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list[float]]:
        """Run the full training loop and return metric history."""
        history: dict[str, list[float]] = {}

        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_one_epoch(train_loader)
            val_metrics = self._validate(val_loader)

            all_metrics = {**{f"train/{k}": v for k, v in train_metrics.items()},
                           **{f"val/{k}": v for k, v in val_metrics.items()}}
            self.logger.log_dict(all_metrics, step=epoch)

            for key, value in all_metrics.items():
                history.setdefault(key, []).append(value)

            primary_metric = val_metrics.get("loss", val_metrics.get("accuracy", 0.0))
            higher_is_better = "accuracy" in val_metrics and "loss" not in val_metrics
            self.checkpoint_mgr.save_best(
                self.model,
                self.optimizer,
                epoch,
                primary_metric,
                filename=self.checkpoint_filename,
                higher_is_better=higher_is_better,
            )

            if self.scheduler is not None:
                self.scheduler.step()

        return history

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _train_one_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Run one training epoch. Return a dict of metric name → value."""

    @abstractmethod
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        """Run one validation pass. Return a dict of metric name → value."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch: Any) -> Any:
        """Recursively move tensors in *batch* to self.device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, (list, tuple)):
            moved = [self._to_device(item) for item in batch]
            return type(batch)(moved)
        return batch
