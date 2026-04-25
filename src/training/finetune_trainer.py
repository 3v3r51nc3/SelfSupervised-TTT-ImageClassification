"""
Trainer for full supervised fine-tuning (Stage B.2).

Both the encoder and the classifier are trained together (no parameter
is frozen). Used as the main downstream stage on CIFAR-10.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import ExperimentLogger


class FineTuneTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        device: torch.device,
        logger: ExperimentLogger,
        checkpoint_mgr: CheckpointManager,
        epochs: int,
        checkpoint_filename: str = "finetune_best.pt",
        use_amp: bool = False,
        label_smoothing: float = 0.0,
        early_stopping_patience: int | None = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            checkpoint_mgr=checkpoint_mgr,
            epochs=epochs,
            checkpoint_filename=checkpoint_filename,
            use_amp=use_amp,
            early_stopping_patience=early_stopping_patience,
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _train_one_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self._backward_step(loss)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with self._autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == targets).sum().item()
                total_samples += batch_size

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }
