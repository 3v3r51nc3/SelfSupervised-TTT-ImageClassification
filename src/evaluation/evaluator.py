"""
Evaluator for clean/corrupted sets.

Minimum metrics:
- Top-1 accuracy,
- delta between without-TTT and with-TTT,
- latency (optional for TER).
"""


class Evaluator:
    # TODO: Evaluate model without TTT.
    def evaluate(self):
        pass

    # TODO: Evaluate model with TTT.
    def evaluate_with_ttt(self):
        pass
