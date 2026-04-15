"""
Trainer for SSL pretraining (SimCLR).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer


class SimCLRTrainer(BaseTrainer):
    def __init__(
        self,
        *args,
        temperature: float,
        log_every_n_steps: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.log_every_n_steps = log_every_n_steps

    def _train_one_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        total_batches = len(loader)

        for batch_idx, batch in enumerate(loader, start=1):
            view_a, view_b = self._extract_views(batch)

            self.optimizer.zero_grad(set_to_none=True)
            z_a, z_b = self.model(view_a, view_b)
            loss = self._nt_xent_loss(z_a, z_b)
            if not torch.isfinite(loss):
                raise ValueError(f"Encountered non-finite training loss at batch {batch_idx}.")

            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())
            self._log_batch_progress(
                stage="train",
                batch_idx=batch_idx,
                total_batches=total_batches,
                view_a=view_a,
                view_b=view_b,
                loss_value=float(loss.item()),
                optimizer_step=True,
            )

        return {"loss": running_loss / max(total_batches, 1)}

    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        total_batches = len(loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                view_a, view_b = self._extract_views(batch)
                z_a, z_b = self.model(view_a, view_b)
                loss = self._nt_xent_loss(z_a, z_b)
                if not torch.isfinite(loss):
                    raise ValueError(f"Encountered non-finite validation loss at batch {batch_idx}.")

                running_loss += float(loss.item())
                self._log_batch_progress(
                    stage="val",
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    view_a=view_a,
                    view_b=view_b,
                    loss_value=float(loss.item()),
                    optimizer_step=False,
                )

        return {"loss": running_loss / max(total_batches, 1)}

    def _extract_views(self, batch: tuple[object, object]) -> tuple[torch.Tensor, torch.Tensor]:
        moved_batch = self._to_device(batch)
        views, _ = moved_batch
        view_a, view_b = views
        return view_a, view_b

    def _nt_xent_loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)

        representations = torch.cat([z_a, z_b], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        batch_size = z_a.size(0)
        mask = torch.eye(2 * batch_size, device=similarity_matrix.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

        positive_logits = torch.cat(
            [
                torch.diag(similarity_matrix, diagonal=batch_size),
                torch.diag(similarity_matrix, diagonal=-batch_size),
            ],
            dim=0,
        )
        log_denominator = torch.logsumexp(similarity_matrix, dim=1)
        return (-positive_logits + log_denominator).mean()

    def _log_batch_progress(
        self,
        stage: str,
        batch_idx: int,
        total_batches: int,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        loss_value: float,
        optimizer_step: bool,
    ) -> None:
        if batch_idx == 1:
            self.logger.debug(
                "%s first batch shapes: view_a=%s view_b=%s batch_size=%d",
                stage,
                tuple(view_a.shape),
                tuple(view_b.shape),
                view_a.size(0),
            )

        if batch_idx % self.log_every_n_steps != 0 and batch_idx != total_batches:
            return

        learning_rate = self.optimizer.param_groups[0]["lr"]
        self.logger.info(
            "%s batch %d/%d - loss=%.6f lr=%.8f",
            stage,
            batch_idx,
            total_batches,
            loss_value,
            learning_rate,
        )
        self.logger.debug(
            "%s batch %d/%d - loss=%.6f finite=%s lr=%.8f optimizer_step=%s",
            stage,
            batch_idx,
            total_batches,
            loss_value,
            math.isfinite(loss_value),
            learning_rate,
            optimizer_step,
        )
