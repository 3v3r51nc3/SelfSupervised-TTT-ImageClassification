"""Pure metric functions for Stage C evaluation."""

from __future__ import annotations

import math

import torch


def per_class_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> tuple[float, ...]:
    """Top-1 accuracy per class.

    Returns a tuple of length ``num_classes``. Classes with zero samples
    yield ``nan`` (defensive — should not happen on a balanced CIFAR-10-C cell).
    """
    if preds.shape != targets.shape:
        raise ValueError(
            f"preds and targets must have the same shape, got {preds.shape} vs {targets.shape}."
        )

    accuracies: list[float] = []
    for cls in range(num_classes):
        mask = targets == cls
        n = int(mask.sum().item())
        if n == 0:
            accuracies.append(math.nan)
            continue
        correct = int((preds[mask] == cls).sum().item())
        accuracies.append(correct / n)
    return tuple(accuracies)
