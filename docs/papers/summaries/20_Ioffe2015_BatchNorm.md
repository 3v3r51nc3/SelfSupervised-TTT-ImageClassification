# Ioffe & Szegedy 2015 — Batch Normalization

**Reference**
Ioffe, S., Szegedy, C.
*Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.*
ICML 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167).

Local PDF: [`BatchNorm_Ioffe2015_1502.03167.pdf`](../pdfs/BatchNorm_Ioffe2015_1502.03167.pdf)

---

## Core idea

Normalizes layer activations **across the batch dimension** to stabilize training:

$$
\hat{x} = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}, \quad y = \gamma \hat{x} + \beta
$$

During training, $\mu, \sigma$ are computed on the current mini-batch. During inference, **running averages** accumulated during training are used (running statistics).

## Historical impact

- Enabled training of much deeper networks (ResNet-50/100/200) by stabilizing gradients.
- Implicit regularization through mini-batch noise.
- Dominated CNN design from 2015 to ~2020.

## Limitation: batch-size sensitivity

- At batch size $\leq 8$, statistics become noisy → instability.
- Poor behavior with non-i.i.d. batches (e.g. 3D detection, full-resolution segmentation).
- This is why LayerNorm, GroupNorm, InstanceNorm have emerged for different domains.

## Why this reference for us (paradoxically)

- **Our ViT does NOT use BatchNorm**, but the BN context is crucial to understand the TTT literature:
  - Sun 2020 (original) uses ResNet + BN.
  - TENT (Wang 2021) adapts only BN affine parameters.
  - The effectiveness of TTT-per-image documented in Sun 2020 partly comes from **BN recalibration** on test-batch statistics.
- Our setup (ViT + LN) **does not benefit** from this mechanism → this is the explanation for the per-image regression.

## Discussion angle

A required reference to explain why per-image works on ResNet (Sun 2020) and fails in our ViT setup.
