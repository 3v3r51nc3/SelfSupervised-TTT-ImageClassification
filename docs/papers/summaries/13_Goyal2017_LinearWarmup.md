# Goyal et al. 2017 — Large Minibatch SGD with Linear Warmup

**Reference**
Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., He, K.
*Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.*
arXiv 2017. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677).

Local PDF: [`LinearWarmup_Goyal2017_1706.02677.pdf`](../pdfs/LinearWarmup_Goyal2017_1706.02677.pdf)

---

## Core idea

Demonstrates that ResNet-50 can be trained on ImageNet in **1 hour** on 256 GPUs with a batch size of 8192. To make this work without losing accuracy, two ingredients are critical:

1. **Linear scaling rule**: when multiplying batch size by $k$, multiply the learning rate by $k$ as well.
2. **Linear warmup**: start with a very small LR and linearly ramp up to the target LR over the first few epochs (typically 5 to 10).

## Why warmup is necessary

At large batch size, the gradient is very precise (low noise), but a randomly initialized network cannot tolerate a large step immediately — risk of explosion. Warmup gives the network time to **calibrate** before the target LR is applied.

## Why this reference for us

- **Linear warmup** is a strict convention for training ViT (which is known to be unstable in the early epochs).
- Our config: `warmup_epochs=10` for SimCLR (200 total epochs, so 5% in warmup), `warmup_epochs=3` for fine-tune (30 epochs).
- Combined with cosine annealing (Loshchilov 2017) for the main phase.

## Note

The approach is now standard in all frameworks (PyTorch Lightning, Hugging Face Trainer, timm) — typically abstracted behind a `lr_scheduler="warmup_cosine"` option.
