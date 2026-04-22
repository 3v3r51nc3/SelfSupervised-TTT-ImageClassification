"""
Fine-tuning Trainer (Stage B.2)

Ce trainer entraîne l’encodeur + le classifieur ensemble.
Contrairement au linear probe, l’encodeur n’est pas gelé.
On optimise donc tous les paramètres du modèle.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.utils.logger import ExperimentLogger
from src.utils.checkpoint import CheckpointManager


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
        )
        self.criterion = nn.CrossEntropyLoss()

    def _train_one_epoch(self, loader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(1) == targets).sum().item()
            total_samples += batch_size

        return {
            "train_loss": total_loss / total_samples,
            "train_accuracy": total_correct / total_samples,
        }

    def _validate(self, loader: DataLoader, epoch: int):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(1) == targets).sum().item()
                total_samples += batch_size

        return {
            "val_loss": total_loss / total_samples,
            "val_accuracy": total_correct / total_samples,
        }
