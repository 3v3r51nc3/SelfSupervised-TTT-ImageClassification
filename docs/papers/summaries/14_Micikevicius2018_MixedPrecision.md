# Micikevicius et al. 2018 — Mixed Precision Training (AMP)

**Reference**
Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., Wu, H.
*Mixed Precision Training.*
ICLR 2018. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740).

Local PDF: [`MixedPrecision_Micikevicius2018_1710.03740.pdf`](../pdfs/MixedPrecision_Micikevicius2018_1710.03740.pdf)

---

## Core idea

Train networks using a **mix of 16-bit (FP16) and 32-bit (FP32)** representations to reduce memory and accelerate computation, without degrading final accuracy.

## Three required techniques

1. **Master weights kept in FP32**. Forward and backward computations run in FP16, but the optimizer updates FP32 copies. Avoids loss of precision on small gradients.
2. **Loss scaling**: multiply the loss by a large factor (typically $2^{16}$) before backward to push small gradients out of the FP16 underflow regime, then divide back afterward.
3. **Selective op casting**: certain ops (softmax, normalization, reductions) stay in FP32 while others (matmul, conv) run in FP16.

## Practical benefits

- **Memory**: roughly 2× less VRAM for activations → enables larger batch sizes.
- **Speed**: on GPUs with Tensor Cores (V100, A100, L4), 2 to 4× speedup on FP16 matmul.
- Final accuracy is **identical** to pure FP32 in most CV / NLP cases.

## Why this reference for us

- **Enabled across the project**: `use_amp: true` in `configs/default.yaml` for Stages A, B.1, B.2.
- Without AMP, SimCLR with batch 1024 does not fit in 22 GB of L4 VRAM.
- Modern implementation: `torch.cuda.amp` (autocast + GradScaler) in `src/training/base_trainer.py`.

## Detail to know

AMP can **interact with TTT**: FP16 gradients are less precise, which may amplify per-image instability. Our TTT code does not use AMP at test time (see `src/ttt/adapter.py`) — a conservative choice to isolate effects.
