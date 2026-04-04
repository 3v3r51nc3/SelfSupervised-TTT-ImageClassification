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
"""

from __future__ import annotations

import argparse

from src.core.config import ConfigLoader
from src.core.pipeline import ExperimentPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SimCLR Stage A training pipeline.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the experiment config YAML.",
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    config = ConfigLoader.load(args.config)
    pipeline = ExperimentPipeline(config=config, config_path=args.config)
    return pipeline.run()


if __name__ == "__main__":
    main()
