"""
Joint trainer for Stage B.2 with the Sun 2020 TTT auxiliary head.

Trains the Y-shape `TTTModel` (encoder + classifier + rotation_head) end
to end. Loss is

    L = CE(classifier(x), y, label_smoothing) + lambda_rot * CE(rotation_head(rotate(x, "rand")))

Validation only reports the supervised CE / top-1 accuracy on the un-rotated
inputs, so it stays directly comparable to `FineTuneTrainer` numbers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.ttt.rotation import rotate_batch
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import ExperimentLogger


class TTTFineTuneTrainer(BaseTrainer):
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
        lambda_rot: float = 1.0,
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
        self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.rot_criterion = nn.CrossEntropyLoss()
        self.lambda_rot = lambda_rot

    def _train_one_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_rot_loss = 0.0
        total_correct = 0
        total_rot_correct = 0
        total_samples = 0

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            rotated, rot_labels = rotate_batch(images, "rand")
            rotated = rotated.to(self.device, non_blocking=True)
            rot_labels = rot_labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                cls_logits = self.model(images)
                rot_logits = self.model.forward_rotation(rotated)
                cls_loss = self.cls_criterion(cls_logits, targets)
                rot_loss = self.rot_criterion(rot_logits, rot_labels)
                loss = cls_loss + self.lambda_rot * rot_loss

            self._backward_step(loss)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_cls_loss += cls_loss.item() * batch_size
            total_rot_loss += rot_loss.item() * batch_size
            total_correct += (cls_logits.argmax(dim=1) == targets).sum().item()
            total_rot_correct += (rot_logits.argmax(dim=1) == rot_labels).sum().item()
            total_samples += batch_size

        return {
            "loss": total_loss / total_samples,
            "cls_loss": total_cls_loss / total_samples,
            "rot_loss": total_rot_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "rot_accuracy": total_rot_correct / total_samples,
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
                    loss = self.cls_criterion(logits, targets)

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == targets).sum().item()
                total_samples += batch_size

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }
