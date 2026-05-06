# Wu & He 2018 — Group Normalization

**Reference**
Wu, Y., He, K.
*Group Normalization.*
ECCV 2018. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494).

Local PDF: [`GroupNorm_Wu2018_1803.08494.pdf`](../pdfs/GroupNorm_Wu2018_1803.08494.pdf)

---

## Core idea

A normalization layer that splits the channels of a feature map into $G$ groups and normalizes within each group, **independently of the batch dimension**. Bridges BatchNorm and LayerNorm:

- $G = 1$ → equivalent to LayerNorm
- $G = C$ (one channel per group) → equivalent to InstanceNorm
- $G$ in between → GroupNorm proper, typically $G = 32$

## Why it exists

BatchNorm fails at small batch sizes (object detection, video, segmentation at full resolution) because batch statistics become noisy. GroupNorm sidesteps the batch entirely while still pooling spatial statistics across multiple channels (unlike LayerNorm which pools all channels).

## Why this reference for us

- Cited in `tech_notes.md` as **background** on normalization choices.
- Not used in our project — ViT's choice of LayerNorm is fixed by the architecture.
- Useful when addressing "why not GroupNorm to get the best of both?": ViT's blocks are designed around LayerNorm; switching norm layers in a Transformer is non-trivial and would diverge from the published recipe.

## For Q&A context

Worth knowing as a third option in the BN/LN/GN landscape — modern hybrid architectures (ConvNeXt, RegNet) sometimes use GroupNorm to combine convolutional inductive bias with batch-size independence.
