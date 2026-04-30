"""
Sun 2020 TTT adapter (rotation auxiliary).

For each (corruption, severity) cell the pipeline calls `reset()` once;
then for each test batch it calls `adapt_and_predict(images)` which:

1. snapshots model + optimizer state if not already snapshotted,
2. for `steps` iterations:
     rotated, rot_labels = rotate_batch(images, rotation_mode)
     rot_logits = model.forward_rotation(rotated)
     loss       = CE(rot_logits, rot_labels)
     loss.backward(); optimizer.step()
3. with no_grad: returns model(images) classification logits.

`reset()` restores the snapshotted state so adaptation never bleeds across
cells.

Adapted from `yueatsprograms/ttt_cifar_release/test_calls/test_adapt.py`
(per-image adapt_single loop) — generalized to per-batch since CIFAR-10-C
evaluation here is batched.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from src.ttt.rotation import rotate_batch


class TestTimeAdapter:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        steps: int = 1,
        learning_rate: float = 1.0e-3,
        adapt_scope: str = "encoder_plus_head",
        rotation_mode: str = "rand",
    ) -> None:
        if steps < 1:
            raise ValueError(f"TTT steps must be >= 1, got {steps}.")
        if adapt_scope not in {"encoder_plus_head"}:
            raise ValueError(
                f"Unsupported adapt_scope {adapt_scope!r}; only 'encoder_plus_head' is supported."
            )
        if rotation_mode not in {"rand", "expand"}:
            raise ValueError(
                f"Unsupported rotation_mode {rotation_mode!r}; expected 'rand' or 'expand'."
            )
        if not (hasattr(model, "encoder") and hasattr(model, "rotation_head")):
            raise ValueError(
                "TestTimeAdapter requires a TTTModel with `encoder` and `rotation_head` attributes."
            )

        self.model = model.to(device)
        self.device = device
        self.steps = steps
        self.rotation_mode = rotation_mode
        self.criterion = nn.CrossEntropyLoss()

        # Adapt encoder + rotation_head; classifier stays frozen so that
        # the supervised head is not perturbed at test time.
        adapt_params = list(self.model.encoder.parameters()) + list(self.model.rotation_head.parameters())
        self.optimizer = torch.optim.SGD(adapt_params, lr=learning_rate)

        self._initial_model_state = deepcopy(self.model.state_dict())
        self._initial_optim_state = deepcopy(self.optimizer.state_dict())

    def reset(self) -> None:
        self.model.load_state_dict(self._initial_model_state)
        self.optimizer.load_state_dict(self._initial_optim_state)

    def adapt_and_predict(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True)

        self.model.train()
        self.model.classifier.eval()
        for p in self.model.classifier.parameters():
            p.requires_grad_(False)

        for _ in range(self.steps):
            rotated, rot_labels = rotate_batch(images, self.rotation_mode)
            rotated = rotated.to(self.device, non_blocking=True)
            rot_labels = rot_labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            rot_logits = self.model.forward_rotation(rotated)
            loss = self.criterion(rot_logits, rot_labels)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            return self.model(images)
