# Szegedy et al. 2016 — Label Smoothing (Inception v3)

**Reference**
Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z.
*Rethinking the Inception Architecture for Computer Vision.*
CVPR 2016. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567).

Local PDF: [`LabelSmoothing_Szegedy2016_InceptionV3_1512.00567.pdf`](../pdfs/LabelSmoothing_Szegedy2016_InceptionV3_1512.00567.pdf)

---

## Core idea (relevant to our project)

The paper introduces the **Inception v3** architecture, but the contribution we use here is **label smoothing** as a regularization technique:

Instead of a one-hot target $y = (0, \dots, 1, \dots, 0)$, use

$$
y' = (1 - \alpha) \cdot y + \frac{\alpha}{K}
$$

where $K$ is the number of classes and $\alpha$ a small coefficient (typically $0.1$).

## Effect

- The network no longer pushes the log-likelihood to $-\infty$ for incorrect classes.
- The logits stay **better calibrated** → softmax probabilities better reflect true confidence.
- Implicit regularization: prevents overfitting by reducing gradient amplitude on "easy" examples.

## Why this reference for us

- **Enabled for fine-tuning** (Stage B.2) with $\alpha = 0.1$. See `configs/default.yaml::finetune.label_smoothing`.
- Particularly useful over 30 epochs on CIFAR-10 where overfitting risk is real.
- Calibration matters for TTT: a poorly calibrated classifier produces noisy entropies that can perturb per-image evaluation.

## Note

PyTorch ships `nn.CrossEntropyLoss(label_smoothing=0.1)` since version 1.10.
