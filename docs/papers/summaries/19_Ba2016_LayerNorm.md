# Ba, Kiros & Hinton 2016 — Layer Normalization

**Reference**
Ba, J. L., Kiros, J. R., Hinton, G. E.
*Layer Normalization.*
arXiv 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450).

Local PDF: [`LayerNorm_Ba2016_1607.06450.pdf`](../pdfs/LayerNorm_Ba2016_1607.06450.pdf)

---

## Core idea

A BatchNorm variant designed for **sequence models** (originally RNNs) where batch size can be small or variable. Instead of normalizing across the batch dimension, it normalizes across the **feature dimension** of a single example:

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$
where $\mu, \sigma$ are computed **per example, across the feature dimension**, independently of the batch.

## Key differences from BatchNorm

| Aspect | BatchNorm | LayerNorm |
|---|---|---|
| Normalizes over | batch dim (per feature) | feature dim (per example) |
| Batch statistics? | yes (running averages in eval) | no (computed on the fly) |
| Sensitive to batch size? | yes (poor at $B = 1$) | no (same at $B = 1$ or $B = 1024$) |
| Train vs eval mode | different | identical |

## Why this reference for us

- **ViT uses LayerNorm** systematically (before each sub-layer: attention and MLP).
- It is **the source of the asymmetry observed** in our per-image vs per-batch results:
  - BatchNorm-based TTT (Sun 2020 original on ResNet) benefits from recalibration of test-batch statistics.
  - LayerNorm has no such mechanism: normalization is per-example, identical in train and test, so TTT cannot exploit that effect.
- See `tech_notes.md::BatchNorm vs LayerNorm` for the detailed derivation.

## Discussion angle

This is the central argument of Annexe C and of the per-image diagnostic. The question "why does per-image degrade in this setup but not in Sun 2020?" necessarily routes through BN vs LN.
