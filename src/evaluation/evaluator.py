"""
Evaluator for clean and corrupted datasets (CIFAR-10 / CIFAR-10-C).

Computes top-1 accuracy, cross-entropy loss, and per-class accuracy
for any DataLoader. Stage C uses evaluate(); Stage D uses evaluate_with_ttt().
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import per_class_accuracy


class TTTAdapter(Protocol):
    def adapt_and_predict(self, images: torch.Tensor) -> torch.Tensor: ...
    def adapt_and_predict_per_image(self, image: torch.Tensor, k: int) -> torch.Tensor: ...
    def reset(self) -> None: ...


class Evaluator:
    def __init__(self, model: nn.Module, device: torch.device, num_classes: int = 10) -> None:
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self, loader: DataLoader) -> dict[str, object]:
        """Evaluate the model without TTT adaptation."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                preds = logits.argmax(dim=1)
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (preds == targets).sum().item()
                total_samples += batch_size

                all_preds.append(preds.detach().cpu())
                all_targets.append(targets.detach().cpu())

        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "per_class_accuracy": per_class_accuracy(preds_cat, targets_cat, self.num_classes),
        }

    def evaluate_per_image_with_ttt(
        self,
        loader: DataLoader,
        ttt_adapter: TTTAdapter,
        k: int,
    ) -> dict[str, object]:
        """
        Sun 2020 per-image TTT: reset the adapter before each image, run K
        replicated rotation steps on that image alone, then classify it.

        Caller is responsible for passing a subsampled loader — per-image
        adaptation on the full 10k cell is wall-clock prohibitive.
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            for i in range(images.size(0)):
                ttt_adapter.reset()
                logits = ttt_adapter.adapt_and_predict_per_image(images[i], k)
                target = targets[i:i + 1]
                loss = self.criterion(logits, target)

                pred = logits.argmax(dim=1)
                total_loss += loss.item()
                total_correct += (pred == target).sum().item()
                total_samples += 1

                all_preds.append(pred.detach().cpu())
                all_targets.append(target.detach().cpu())

        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "per_class_accuracy": per_class_accuracy(preds_cat, targets_cat, self.num_classes),
        }

    def evaluate_with_ttt(self, loader: DataLoader, ttt_adapter: TTTAdapter) -> dict[str, object]:
        """Evaluate the model with TTT — adapt then predict per batch."""
        ttt_adapter.reset()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = ttt_adapter.adapt_and_predict(images)
            loss = self.criterion(logits, targets)

            preds = logits.argmax(dim=1)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())

        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "per_class_accuracy": per_class_accuracy(preds_cat, targets_cat, self.num_classes),
        }
