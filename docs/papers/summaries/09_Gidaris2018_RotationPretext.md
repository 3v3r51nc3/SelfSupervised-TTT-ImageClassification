# Gidaris et al. 2018 — Rotation Prediction Pretext

**Reference**
Gidaris, S., Singh, P., Komodakis, N.
*Unsupervised Representation Learning by Predicting Image Rotations.*
ICLR 2018. [arXiv:1803.07728](https://arxiv.org/abs/1803.07728).

Local PDF: [`Rotation_Gidaris2018_1803.07728.pdf`](../pdfs/Rotation_Gidaris2018_1803.07728.pdf)

---

## Core idea

Extremely simple self-supervised pretext task:

1. Given an image, randomly apply a rotation from $\{0°, 90°, 180°, 270°\}$.
2. The synthetic label is the **rotation index** (4-class problem).
3. Train the network (ResNet-CIFAR or larger) to predict the index using cross-entropy.

To solve this, the network must learn to recognize **canonical orientations** of objects — faces upright, cars driving horizontally, etc. — which forces semantically rich representations.

## Why it works

- Well-posed task: natural images have a canonical orientation, so the signal is non-trivial.
- The network cannot cheat with low-level artifacts (rotation preserves color statistics, etc.).
- No need for image pairs (unlike contrastive learning) → cheap.

## Main results

- Linear probe on ImageNet ≈ 50% top-1 — not at SimCLR/MoCo level, but very competitive for 2018.
- More importantly, serves as a **building block** in Sun 2020 TTT as the auxiliary test-time task.

## Why this reference for us

- This is the **pretext task used by Sun 2020 TTT**, hence indirectly used in our Stage B.2 and Stage C.
- See `src/ttt/rotation.py` for our `rotate_batch()` implementation.
- Justifies why rotation is a sensible "default" pretext task: simple, few hyperparameters, works on natural images.

## Discussion angle

On the question of alternative pretext tasks (jigsaw, colorization, masked autoencoding): Sun 2020 specifically uses rotation, which is what the subject demands. Other pretexts are a natural extension for the final report.
