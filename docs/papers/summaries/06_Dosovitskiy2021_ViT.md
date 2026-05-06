# Dosovitskiy et al. 2021 — Vision Transformer (ViT)

**Reference**
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.
*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.*
ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929).

Local PDF: [`ViT_Dosovitskiy2021_2010.11929.pdf`](../pdfs/ViT_Dosovitskiy2021_2010.11929.pdf)

---

## Core idea

Applies the **Transformer** architecture (originally from NLP) to images by treating them as a sequence of patches:

1. Split the image into non-overlapping fixed-size patches (e.g. $16 \times 16$, or $4 \times 4$ for CIFAR).
2. Flatten and project each patch to an embedding (linear projection).
3. Add a learned **\[CLS] token** and **positional embeddings**.
4. Pass the sequence through a standard Transformer encoder (multi-head self-attention + MLP, normalized via **LayerNorm**).
5. Use the final \[CLS] representation for classification.

## Main contribution

Demonstrates that a pure Transformer, **without convolutional inductive biases** (no built-in locality, no enforced translation equivariance), can match or beat CNNs when given enough data.

## Variants by size

| Model | Layers | Embed dim | Heads | Params |
|---|---|---|---|---|
| ViT-Tiny | 12 | 192 | 3 | 5.7M |
| ViT-Small | 12 | 384 | 6 | 22M |
| ViT-Base | 12 | 768 | 12 | 86M |
| ViT-Large | 24 | 1024 | 16 | 307M |

## Why this reference for us

- It is our **backbone** (Stages A and B). See `src/models/backbone.py`.
- Explicitly required by the TER subject: "ViT as encoder".
- The presence of **LayerNorm** (instead of BatchNorm in CNNs) has a major effect on TTT behavior, especially per-image — this is the principal source of the asymmetry seen in our results.

## Configuration we use

- ViT-Tiny adapted for CIFAR: patch $4 \times 4$, embed dim 192, 12 layers, 3 heads.
- Drop_path 0.1 for regularization (cf. Stochastic Depth).
- Implementation via the [`timm`](https://github.com/huggingface/pytorch-image-models) library.
