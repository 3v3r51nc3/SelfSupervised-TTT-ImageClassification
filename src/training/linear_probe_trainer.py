"""
Linear Probe Trainer (Stage B.1)

Ce module implémente l’entraînement supervisé d’un classifieur linéaire
sur l’encodeur pré‑entraîné par SimCLR (Stage A). L’objectif du linear probe
est d’évaluer la qualité des représentations apprises en auto‑supervisé :
l’encodeur est gelé et seule la couche linéaire est entraînée.

Ce trainer hérite de BaseTrainer et fournit les deux méthodes essentielles :
- _train_one_epoch : boucle d’entraînement supervisé (CrossEntropy + accuracy)
- _validate        : boucle de validation sans gradient

Le modèle passé au trainer doit être de la forme :
    FineTuneModel(encoder_frozen, LinearClassifier)

Ce stage permet de mesurer la performance du backbone SSL sans fine‑tuning,
et sert de baseline avant Stage B.2 (full fine‑tuning).
"""


from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.utils.logger import ExperimentLogger
from src.utils.checkpoint import CheckpointManager



class LinearProbeTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        device: torch.device,
        logger: ExperimentLogger,
        checkpoint_mgr: CheckpointManager,
        epochs: int,
        checkpoint_filename: str = "linear_probe_best.pt",
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
        
    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)          # forward
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
        }

    def _validate(self, loader: DataLoader, epoch: int) -> dict[str, float]:
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

                preds = logits.argmax(dim=1)
                total_correct += (preds == targets).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
        }

