"""
Sun 2020 TTT adapter (rotation auxiliary).

Two adaptation modes:

- `adapt_and_predict(images)` — per-batch. One SGD step on the rotation
  loss for the whole batch, then classify. Used as the operational
  baseline.
- `adapt_and_predict_per_image(image, k)` — per-image. Replicate one
  image K times, draw K random rotations, one SGD step on the K-batch,
  then classify the original image. Matches TER2.pdf "individuellement
  à chaque image".

For each (corruption, severity) the pipeline must call `reset()` to
restore the snapshotted clean weights so adaptation never bleeds across
cells; for per-image the caller resets between every image.

Adapted from `yueatsprograms/ttt_cifar_release/test_calls/test_adapt.py`.
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
        self._set_train_mode_with_classifier_frozen()

        for _ in range(self.steps):
            self._rotation_step(images)

        self.model.eval()
        with torch.no_grad():
            return self.model(images)

    def adapt_and_predict_per_image(self, image: torch.Tensor, k: int) -> torch.Tensor:
        """
        Sun 2020 per-image TTT: replicate the single test image into K
        copies, draw K random rotation labels, take `steps` SGD steps on
        the K-batch, then classify the original image.

        Caller must `reset()` before each image so adaptation does not
        leak across samples.
        """
        if k < 1:
            raise ValueError(f"per-image augmentation count k must be >= 1, got {k}.")
        if image.dim() == 3:
            image = image.unsqueeze(0)
        elif image.dim() != 4 or image.size(0) != 1:
            raise ValueError(
                f"adapt_and_predict_per_image expects a single image (CHW or 1xCxHxW), got shape {tuple(image.shape)}."
            )
        image = image.to(self.device, non_blocking=True)

        self._set_train_mode_with_classifier_frozen()

        replicated = image.expand(k, -1, -1, -1)
        for _ in range(self.steps):
            self._rotation_step(replicated)

        self.model.eval()
        with torch.no_grad():
            return self.model(image)

    def _set_train_mode_with_classifier_frozen(self) -> None:
        self.model.train()
        self.model.classifier.eval()
        for p in self.model.classifier.parameters():
            p.requires_grad_(False)

    def _rotation_step(self, batch: torch.Tensor) -> None:
        rotated, rot_labels = rotate_batch(batch, self.rotation_mode)
        rotated = rotated.to(self.device, non_blocking=True)
        rot_labels = rot_labels.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        rot_logits = self.model.forward_rotation(rotated)
        loss = self.criterion(rot_logits, rot_labels)
        loss.backward()
        self.optimizer.step()
