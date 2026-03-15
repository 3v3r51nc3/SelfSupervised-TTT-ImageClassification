"""
Base trainer.

Contains shared loop structure:
- train loop,
- validation loop,
- checkpoint/log hooks.
"""


class BaseTrainer:
    # TODO: Shared utility methods for all trainers.
    def fit(self) -> None:
        pass
