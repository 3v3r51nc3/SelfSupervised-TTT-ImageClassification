"""
Trainer for full fine-tuning.
"""

from src.training.base_trainer import BaseTrainer


class FineTuneTrainer(BaseTrainer):
    # TODO: Unfreeze encoder and run supervised fine-tuning.
    def fit(self) -> None:
        pass
