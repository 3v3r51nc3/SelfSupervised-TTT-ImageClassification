# Loshchilov & Hutter 2019 — AdamW

**Reference**
Loshchilov, I., Hutter, F.
*Decoupled Weight Decay Regularization.*
ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).

Local PDF: [`AdamW_Loshchilov2019_1711.05101.pdf`](../pdfs/AdamW_Loshchilov2019_1711.05101.pdf)

---

## Core idea

The classical `weight_decay` of Adam is **mathematically equivalent to L2 regularization** only for plain SGD. For adaptive optimizers (Adam, RMSprop), L2 and weight decay **are no longer the same**: regularization is divided by the running gradient mean, which dampens its effect on parameters with large gradients.

AdamW fixes this by **decoupling**:

- the gradient update (pure Adam),
- the weight decay (applied separately, SGD-style).

$$
\theta_{t+1} = \theta_t - \eta \cdot \hat m_t / (\sqrt{\hat v_t} + \epsilon) - \eta \cdot \lambda \cdot \theta_t
$$

## Practical effect

- Regularization is now uniformly applied across all parameters.
- Often gives better generalization than vanilla Adam, especially on Transformers.
- The `weight_decay` hyperparameter must be retuned (typically larger than for Adam) — usually $0.01$ to $0.1$.

## Why this reference for us

- **Optimizer used** in Stage A (SimCLR), B.1 (linear probe) and B.2 (fine-tune). See `configs/default.yaml`.
- Specifically recommended for ViT (the ViT paper itself uses AdamW).
- Our config: `lr=5e-4`, `weight_decay=0.05` for fine-tune (Stage B.2).

## Note

PyTorch ships `torch.optim.AdamW` which directly implements this variant.
