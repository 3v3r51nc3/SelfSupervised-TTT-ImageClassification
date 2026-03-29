"""
Checkpoint save/load helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CheckpointManager:
    """Save and load model + optimizer state dicts."""

    def __init__(self, checkpoint_dir: str, save_best_only: bool = True) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._save_best_only = save_best_only
        self._best_metric: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric: float,
        filename: str,
    ) -> Path:
        """Unconditionally save a checkpoint and return its path."""
        path = self._dir / filename
        torch.save(
            {
                "epoch": epoch,
                "metric": metric,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )
        return path

    def save_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric: float,
        filename: str = "best.pt",
        higher_is_better: bool = True,
    ) -> bool:
        """Save only when *metric* improves. Returns True if saved."""
        is_better = (
            self._best_metric is None
            or (higher_is_better and metric > self._best_metric)
            or (not higher_is_better and metric < self._best_metric)
        )
        if not self._save_best_only or is_better:
            self._best_metric = metric
            self.save(model, optimizer, epoch, metric, filename)
            return True
        return False

    def load(self, filename: str, map_location: Any = "cpu") -> dict[str, Any]:
        """Load a checkpoint dict. Caller applies state dicts."""
        path = self._dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location=map_location, weights_only=True)

    def load_model(
        self,
        model: nn.Module,
        filename: str,
        map_location: Any = "cpu",
    ) -> int:
        """Load state dict into *model* and return the saved epoch."""
        ckpt = self.load(filename, map_location=map_location)
        model.load_state_dict(ckpt["model_state_dict"])
        return int(ckpt["epoch"])
