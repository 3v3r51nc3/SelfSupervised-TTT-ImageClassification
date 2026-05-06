# Cubuk et al. 2020 — RandAugment

**Reference**
Cubuk, E. D., Zoph, B., Shlens, J., Le, Q. V.
*RandAugment: Practical Automated Data Augmentation with a Reduced Search Space.*
NeurIPS 2020. [arXiv:1909.13719](https://arxiv.org/abs/1909.13719).

Local PDF: [`RandAugment_Cubuk2020_1909.13719.pdf`](../pdfs/RandAugment_Cubuk2020_1909.13719.pdf)

---

## Core idea

A simplified data-augmentation strategy that replaces hand-tuned pipelines (such as AutoAugment, which requires costly RL search). Only two hyperparameters:

- $N$: **number** of augmentations applied per image.
- $M$: a single **magnitude** for all transformations (scale 0 to 30).

For each image, $N$ transformations are **randomly drawn** from a fixed pool of 14 (rotation, shear, translate, color, contrast, brightness, sharpness, solarize, equalize, etc.) and applied with magnitude $M$.

## Advantages

- No per-dataset augmentation search needed (AutoAugment requires search per dataset, RandAugment does not).
- Performance equivalent to AutoAugment on ImageNet, ResNet-50, EfficientNet.
- Simple to implement, fast to tune ($N \in \{1, 2, 3\}$, $M \in \{5, 10, 15, 20\}$).

## Why this reference for us

- **Augmentation used during supervised fine-tuning** (Stage B.2): `randaugment_n=2, randaugment_m=9` in `configs/default.yaml::data`.
- Increases classifier robustness **without ever seeing CIFAR-10-C during training**.
- Important distinction: SimCLR augmentations (Stage A) are different (stronger, contrastive). RandAugment is exclusively for Stage B.2.

## Standard ViT augmentation trio

Modern ViT training typically uses:

1. RandAugment (this paper),
2. Random Erasing (Zhong 2020),
3. Mixup / CutMix (not used in this project to stay close to Sun 2020).
