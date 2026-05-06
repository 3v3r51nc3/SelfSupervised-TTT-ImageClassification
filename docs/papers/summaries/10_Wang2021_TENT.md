# Wang et al. 2021 — TENT (Test-Time Entropy Minimization)

**Reference**
Wang, D., Shelhamer, E., Liu, S., Olshausen, B., Darrell, T.
*Tent: Fully Test-Time Adaptation by Entropy Minimization.*
ICLR 2021. [arXiv:2006.10726](https://arxiv.org/abs/2006.10726).

Local PDF: [`TENT_Wang2021_2006.10726.pdf`](../pdfs/TENT_Wang2021_2006.10726.pdf)

---

## Core idea

Adapt a model at test time **without an auxiliary pretext task**. The adaptation loss is the **entropy of the predictions** themselves:

$$
\mathcal{L}_{\text{TENT}} = -\sum_y p(y \mid x) \log p(y \mid x)
$$

Minimizing this entropy pushes the model toward confident (one-hot) predictions, which effectively **sharpens decision boundaries** on the test distribution.

## Critical design choice

TENT updates only the **affine parameters of BatchNorm** (`gamma`, `beta`) — this is:

- **lightweight** (very few parameters),
- **fast** (1 backward = 1 forward),
- but **strictly tied to BatchNorm-based architectures**.

It does not work as-is on ViT, which uses LayerNorm.

## TENT vs Sun 2020 TTT

| Aspect | TENT | Sun 2020 |
|---|---|---|
| Test-time loss | none (entropy of logits) | rotation auxiliary |
| What is adapted | BN affine params | encoder + rotation head |
| Special pretraining? | no | yes (auxiliary head) |
| Works on ViT/LN? | no (without modification) | yes |

## Why this reference for us

- Cited in `tech_notes.md` as a **method we considered but rejected**: TENT would require either adding BN-style heads to ViT or switching to ResNet, which would deviate from the TER subject.
- Important related-work reference: it is the most popular TTA baseline of 2021.

## Discussion angle

On the question "why not TENT?":

> *"TENT is designed for BatchNorm — it modifies the affine parameters of BN layers, which do not exist in ViT. Using it would require either switching to ResNet (which the subject does not allow) or modifying ViT to introduce BN layers, which would denature the backbone."*
