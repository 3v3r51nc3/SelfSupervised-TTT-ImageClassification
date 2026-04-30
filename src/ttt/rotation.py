"""
Rotation utilities for Sun 2020 TTT.

Module goals:
- rotate single CHW tensors by 0/90/180/270 degrees,
- batch-rotate either with explicit per-sample labels, with random labels,
  or with the "expand" mode (each input is expanded into all 4 rotations).

Adapted from `yueatsprograms/ttt_cifar_release/utils/rotation.py`.
"""

from __future__ import annotations

import torch

# Tensors are assumed to be (C, H, W) for the single-image helpers and
# (N, C, H, W) for the batch helpers. Labels are in {0, 1, 2, 3} meaning
# {0°, 90°, 180°, 270°} respectively — same convention as the reference repo.


def tensor_rot_90(x: torch.Tensor) -> torch.Tensor:
    return x.flip(2).transpose(1, 2)


def tensor_rot_180(x: torch.Tensor) -> torch.Tensor:
    return x.flip(2).flip(1)


def tensor_rot_270(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).flip(2)


def rotate_batch_with_labels(batch: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Apply per-sample rotation given labels in {0, 1, 2, 3}."""
    images = []
    for img, label in zip(batch, labels):
        label_int = int(label.item())
        if label_int == 0:
            rotated = img
        elif label_int == 1:
            rotated = tensor_rot_90(img)
        elif label_int == 2:
            rotated = tensor_rot_180(img)
        elif label_int == 3:
            rotated = tensor_rot_270(img)
        else:
            raise ValueError(f"Rotation label must be in {{0, 1, 2, 3}}, got {label_int}.")
        images.append(rotated.unsqueeze(0))
    return torch.cat(images)


def rotate_batch(batch: torch.Tensor, mode: str | int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate a batch with one of three modes:
    - "rand"   : each sample gets a random label in {0, 1, 2, 3}.
    - "expand" : returns 4 * len(batch) samples — every input rotated by
                 every angle. Labels concatenated as [0]*N + [1]*N + ...
    - int      : every sample gets the given label.
    Returns (rotated_batch, labels).
    """
    if mode == "rand":
        labels = torch.randint(4, (len(batch),), dtype=torch.long)
    elif mode == "expand":
        labels = torch.cat(
            [
                torch.zeros(len(batch), dtype=torch.long),
                torch.zeros(len(batch), dtype=torch.long) + 1,
                torch.zeros(len(batch), dtype=torch.long) + 2,
                torch.zeros(len(batch), dtype=torch.long) + 3,
            ]
        )
        batch = batch.repeat((4, 1, 1, 1))
    elif isinstance(mode, int):
        labels = torch.zeros((len(batch),), dtype=torch.long) + mode
    else:
        raise ValueError(f"rotate_batch mode must be 'rand', 'expand', or int; got {mode!r}.")
    return rotate_batch_with_labels(batch, labels), labels
