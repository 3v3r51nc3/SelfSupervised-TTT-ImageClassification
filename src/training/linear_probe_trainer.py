"""
Trainer for linear probe.
"""

from src.training.base_trainer import BaseTrainer


class LinearProbeTrainer(BaseTrainer):
    # TODO: Train only the linear head with frozen encoder.
    def fit(self) -> None:
        pass
