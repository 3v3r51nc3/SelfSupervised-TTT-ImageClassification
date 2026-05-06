# Chen et al. 2020 — SimCLR

**Reference**
Chen, T., Kornblith, S., Norouzi, M., Hinton, G.
*A Simple Framework for Contrastive Learning of Visual Representations.*
ICML 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709).

Local PDF: [`SimCLR_Chen2020_2002.05709.pdf`](../pdfs/SimCLR_Chen2020_2002.05709.pdf)

---

## Core idea

Learns image representations **without labels** through contrastive learning:

1. From an image $x$, generate two **strongly augmented views** $\tilde x_1, \tilde x_2$ (random crop, color jitter, blur, gray).
2. An encoder $f$ (ResNet or ViT) produces features, then a **projection head** $g$ produces embeddings $z_1, z_2$.
3. An **NT-Xent** (Normalized Temperature-scaled Cross-Entropy) loss pulls $z_1, z_2$ together and pushes apart all other (negative) pairs in the batch.

## Four key ingredients (per the authors)

1. **Strong, composed augmentations** (the dominant factor).
2. **Non-linear projection head** (MLP) — better than a linear head; discarded after pretraining.
3. **Large batch size** (rich contrastive signal from many negatives).
4. **Long training** (more is better).

## NT-Xent loss

$$
\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1, k \neq i}^{2N} \exp(\text{sim}(z_i, z_k) / \tau)}
$$
where $\text{sim}(\cdot, \cdot)$ is cosine similarity and $\tau$ is the temperature.

## Main results

- **Linear probe on ImageNet**: 76.5% top-1 with ResNet-50, comparable to supervised pretraining.
- Confirmed contrastive representations are competitive without labels.

## Why this reference for us

- **Stage A of our pipeline** — exactly the pretraining method. See `src/training/simclr_trainer.py` and `src/models/simclr.py`.
- The projection head is discarded after Stage A (per the original paper) — this is why it is not reused during TTT.

## Configuration we use

- 200 epochs, batch 1024, $\tau = 0.2$.
- Backbone: ViT-Tiny (instead of original ResNet).
- Projection: MLP → 128-d.
- Augmentations: crop + horizontal flip + color jitter + gray + blur.
