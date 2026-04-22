"""
Evaluator for clean and corrupted datasets (CIFAR-10 / CIFAR-10-C).

Responsibilities:
- Compute Top-1 accuracy and loss on any DataLoader,
- Provide a unified interface for evaluation,
- Support both standard evaluation (no TTT) and TTT-based evaluation.

Stage C uses only evaluate().
Stage D will use evaluate_with_ttt().
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Evaluator:
    def __init__(self, model: nn.Module, device: torch.device, logger):
        """
        model  : trained model (fine-tuned encoder + classifier)
        device : CPU/GPU
        logger : ExperimentLogger
        """
        self.model = model
        self.device = device
        self.logger = logger
        self.criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # Standard evaluation (used for Stage C: CIFAR-10-C)
    # ---------------------------------------------------------
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """
        Evaluate the model without TTT adaptation.
        Returns:
            - loss
            - accuracy
        """
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

                total_loss += loss.item() * images.size(0)
                total_correct += (logits.argmax(1) == targets).sum().item()
                total_samples += images.size(0)

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    # ---------------------------------------------------------
    # Evaluation with TTT (Stage D)
    # ---------------------------------------------------------
    def evaluate_with_ttt(self, loader: DataLoader, ttt_adapter) -> dict[str, float]:
        """
        Evaluate the model with Test-Time Training (TTT).
        ttt_adapter must implement:
            - adapt_step(images)
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0

        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Adaptation step(s)
            ttt_adapter.adapt_step(images)

            # Prediction after adaptation
            with torch.no_grad():
                logits = self.model(images)
                preds = logits.argmax(1)

            total_correct += (preds == targets).sum().item()
            total_samples += images.size(0)

        return {
            "accuracy": total_correct / total_samples,
        }
