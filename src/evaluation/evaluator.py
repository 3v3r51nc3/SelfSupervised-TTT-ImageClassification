"""
Evaluator for clean and corrupted datasets (CIFAR-10 / CIFAR-10-C).

Computes top-1 accuracy and cross-entropy loss for any DataLoader.
Stage C uses evaluate(); Stage D will use evaluate_with_ttt().
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class TTTAdapter(Protocol):
    def adapt_step(self, images: torch.Tensor) -> None: ...


class Evaluator:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate the model without TTT adaptation."""
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
                total_correct += (logits.argmax(dim=1) == targets).sum().item()
                total_samples += batch_size

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def evaluate_with_ttt(self, loader: DataLoader, ttt_adapter: TTTAdapter) -> dict[str, float]:
        """Evaluate the model with Test-Time Training (Stage D)."""
        self.model.eval()
        total_correct = 0
        total_samples = 0

        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            ttt_adapter.adapt_step(images)

            with torch.no_grad():
                logits = self.model(images)

            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += images.size(0)

        return {"accuracy": total_correct / total_samples}
