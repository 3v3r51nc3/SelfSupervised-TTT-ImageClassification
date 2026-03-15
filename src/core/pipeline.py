"""
Experiment orchestrator.

ExperimentPipeline should:
- wire all dependencies (data/model/trainers/evaluator),
- run stages in the correct order,
- control artifact persistence.
"""


class ExperimentPipeline:
    # TODO: Accept ExperimentConfig and initialize all components.
    def __init__(self) -> None:
        pass

    # TODO: Run full pipeline: pretrain -> probe/fine-tune -> TTT eval.
    def run(self) -> None:
        pass
