# Loshchilov & Hutter 2017 — SGDR (Cosine Annealing)

**Reference**
Loshchilov, I., Hutter, F.
*SGDR: Stochastic Gradient Descent with Warm Restarts.*
ICLR 2017. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983).

Local PDF: [`SGDR_Loshchilov2017_1608.03983.pdf`](../pdfs/SGDR_Loshchilov2017_1608.03983.pdf)

---

## Core idea

Proposes a **learning-rate schedule** that follows a **cosine** curve between a maximum $\eta_{\max}$ and a minimum $\eta_{\min}$ over the duration of training, optionally with **warm restarts** (resetting back to $\eta_{\max}$ to explore multiple minima).

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\!\left(\frac{T_{\text{cur}}}{T_i} \pi\right)\right)
$$

## Practical effect

- LR starts high (exploration), gently decreases at the end (refinement).
- The cosine shape is smoother than step decay, and empirically converges better.
- Warm restarts allow you to **ensemble** several local minima without training multiple models.

## Standard combo: warmup + cosine

A widespread practice since 2019 (notably for ViT, BERT, SimCLR):

1. **Linear warmup** for $W$ epochs (LR ramps from 0 to $\eta_{\max}$).
2. **Cosine annealing** over the remaining $T - W$ epochs.

This combination stabilizes the early-training phase (important for Transformers) while keeping the smooth convergence of cosine.

## Why this reference for us

- **Scheduler used** in all training stages (A, B.1, B.2).
- See the scheduler factory in `src/training/base_trainer.py`.
- Configuration: `warmup_epochs=10` for SimCLR, then cosine over the remaining 190 epochs.
