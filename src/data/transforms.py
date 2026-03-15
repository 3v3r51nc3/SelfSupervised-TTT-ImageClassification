"""
Augmentation factory.

Separate modes are required:
- simclr_aug: two strong views of one image,
- eval_aug: minimal processing for evaluation.
"""


class TransformFactory:
    # TODO: Build transforms for SimCLR.
    def build_simclr(self):
        pass

    # TODO: Build transforms for eval/fine-tune.
    def build_eval(self):
        pass
