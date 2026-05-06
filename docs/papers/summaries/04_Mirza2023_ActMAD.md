# Mirza et al. 2023 — ActMAD

**Reference**
Mirza, M. J., Jané Soneira, P., Lin, W., Kozinski, M., Possegger, H., Bischof, H.
*ActMAD: Activation Matching to Align Distributions for Test-Time-Training.*
CVPR 2023.

Local PDF: [`Actmad_Activation_matching_to_align_distributions_for_test_time_training_compressed.pdf`](../pdfs/Actmad_Activation_matching_to_align_distributions_for_test_time_training_compressed.pdf)

---

## Core idea

Instead of using a **pretext task** (rotation as in Sun 2020, contrastive as in TTT++), ActMAD directly aligns the **distribution of intermediate activations** of the network at test time with the distribution seen during training. The criterion is a *Maximum Mean Discrepancy* (MMD) between activation statistics.

## Claimed advantages

- No auxiliary head needs to be pretrained → applies to any existing backbone.
- Fast adaptation (no head to update, only activation statistics).
- Modality-agnostic (vision in the paper, but also 3D detection).

## Why this reference for us

- **Fourth reference in the TER subject** ([4]) — figure from this paper is included in the subject.
- Represents an alternative TTT family (distribution alignment) we can cite when contrasting Sun 2020 (rotation) and TTT++ (contrastive).

## Connection to our results

If we wanted to push further, ActMAD would give us a **baseline without a pretext task** — a useful comparison point to measure what rotation actually contributes. It is a natural extension in the "Next steps" section of the final report.

## Discussion angle

A natural answer to "have you considered other TTT methods?" — ActMAD is a different-family baseline (no pretext task), suitable to include in the final report.
