# Liu et al. 2021 — TTT++ (When Does SSL TTT Fail or Thrive?)

**Reference**
Liu, Y., Kothari, P., Van Delft, B., Bellot-Gurlet, B., Mordan, T., Alahi, A.
*TTT++: When does self-supervised test-time training fail or thrive?*
NeurIPS 2021. [arXiv:2106.10802](https://arxiv.org/abs/2106.10802).

Local PDF: [`TTT_When_does_self_supervised_test_time_training_fail_or_thrive.pdf`](../pdfs/TTT_When_does_self_supervised_test_time_training_fail_or_thrive.pdf)

---

## Core idea

Improves Sun 2020 TTT along two axes:

1. **Contrastive pretext task** (NT-Xent, SimCLR-style) instead of rotation. The contrastive representation captures more structure, so each adaptation step is more informative.
2. **Feature alignment regularization**: during testing, the statistics (mean / covariance) of the encoder's features are constrained to stay close to those seen during training, preventing drift.

## "When does TTT thrive" diagnostic

The paper identifies regimes where TTT works or fails:

- **Works well**: large distribution shift, well-pretrained backbone, informative SSL loss.
- **Fails**: small shift (cost of adaptation exceeds the gain), weak SSL signal, test distribution very far from pretraining distribution.

## Why this reference for us

- It is the **second reference of the TER subject** ([2]).
- Relevant to explaining **why we do not use NT-Xent at test time** in Stage C (we follow Sun 2020, not TTT++). See our Annexe K.
- Provides a theoretical framework for our results: per-batch works (TTT thrives), per-image degrades (TTT fails) — the regimes are symmetric to those analyzed in this paper.

## Connection to our observations

Our regressing per-image results align with the "thin SSL signal" scenario: 128 rotated copies of a single image carry less information than a true contrastive batch.

## Possible extension

Implementing the NT-Xent variant at test time (instead of rotation) would be a natural Sun 2020 vs TTT++ comparison in our ViT setup.
