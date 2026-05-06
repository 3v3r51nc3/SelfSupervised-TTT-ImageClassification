# Huang et al. 2016 — Stochastic Depth (Drop Path)

**Reference**
Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K.
*Deep Networks with Stochastic Depth.*
ECCV 2016. [arXiv:1603.09382](https://arxiv.org/abs/1603.09382).

Local PDF: [`StochasticDepth_Huang2016_1603.09382.pdf`](../pdfs/StochasticDepth_Huang2016_1603.09382.pdf)

---

## Core idea

Regularization for residual networks: during training, **randomly drop entire residual blocks** with a probability that increases with depth. At inference, all blocks are used (with proper rescaling).

At each step:
$$
y_{\ell+1} = b_{\ell} \cdot f_{\ell}(y_{\ell}) + y_{\ell}
$$
where $b_{\ell} \sim \text{Bernoulli}(p_{\ell})$ and $p_{\ell}$ decays linearly from 1 (early layers) to $1 - p_L$ (depth $L$).

## Effect

- **Reduces overfitting** in deeper layers.
- **Speeds up training** (less effective compute) without degrading final accuracy.
- Implicitly trains an **ensemble** of varying-depth subnetworks.

## Modern variant: DropPath

For Transformers, the term used is **DropPath**: an entire Transformer block (attention + MLP) is dropped per sample in the batch. The mechanism is identical, only the terminology changes.

## Why this reference for us

- **Enabled on ViT-Tiny**: `drop_path_rate=0.1` in `configs/default.yaml::model`.
- Standard regularization for Transformer training (ViT, Swin, BEiT all use DropPath).
- Without it, ViT-Tiny shows late-stage overfitting signs across 200 SimCLR epochs + 30 fine-tune epochs.
