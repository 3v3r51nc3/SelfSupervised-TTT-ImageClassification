# Paszke et al. 2019 — PyTorch

**Reference**
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.
*PyTorch: An Imperative Style, High-Performance Deep Learning Library.*
NeurIPS 2019. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703).

Local PDF: [`PyTorch_Paszke2019_1912.01703.pdf`](../pdfs/PyTorch_Paszke2019_1912.01703.pdf)

---

## Core idea

The official paper describing the design of the PyTorch deep-learning framework. Three main design choices:

1. **Imperative ("define-by-run") graph construction** — operations execute as they are called, debugging is identical to plain Python.
2. **GPU/CPU agnostic tensor library** with autograd built in.
3. **Pythonic APIs** for layers, datasets, optimizers — no domain-specific language to learn.

## Why this reference for us

- **Framework used for the entire project**: SimCLR trainer, ViT model, TTT adapter, evaluation loop, all in PyTorch.
- Pulls in everything we depend on: `torch.nn`, `torch.optim.AdamW`, `torch.cuda.amp`, `torch.utils.data.DataLoader`.
- The paper is cited for academic completeness — most ML papers in 2024+ cite it because their experiments rely on it.

## Version we use

PyTorch ≥ 2.x with CUDA support. Specific deps in `requirements.txt`. AMP and `torch.compile` are available but `torch.compile` is **not** used in this project (kept off to ensure deterministic gradient flow during TTT).
