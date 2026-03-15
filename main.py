"""
Project entry point.

PIPELINE (SimCLR + ViT + CIFAR-10 + TTT):
1) Load experiment config and fix random seed.
2) Prepare CIFAR-10 splits:
   - train_ssl/val for pretraining and validation,
   - test for final evaluation.
3) Stage A: self-supervised pretraining (SimCLR) on ViT encoder.
4) Stage B: downstream evaluation:
   - linear probe (frozen encoder),
   - full fine-tune (unfrozen encoder).
5) Stage C: test-time training (TTT):
   - adapt on test stream with a self-supervised objective,
   - predict after K adaptation steps.
6) Save metrics, logs, and checkpoints.

This file is a skeleton only: structure and comments, no implementation yet.
"""

from src.core.pipeline import ExperimentPipeline


def main() -> None:
    # TODO: Initialize ExperimentPipeline from config.
    # TODO: Run stages in order: pretrain -> probe/fine-tune -> TTT eval.
    # TODO: Persist final metrics to artifacts/ or logs/.
    pipeline = ExperimentPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
