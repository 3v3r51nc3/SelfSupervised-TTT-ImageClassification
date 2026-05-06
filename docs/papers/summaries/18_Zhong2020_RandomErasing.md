# Zhong et al. 2020 — Random Erasing

**Reference**
Zhong, Z., Zheng, L., Kang, G., Li, S., Yang, Y.
*Random Erasing Data Augmentation.*
AAAI 2020. [arXiv:1708.04896](https://arxiv.org/abs/1708.04896).

Local PDF: [`RandomErasing_Zhong2020_1708.04896.pdf`](../pdfs/RandomErasing_Zhong2020_1708.04896.pdf)

---

## Core idea

Simple augmentation that **masks a random rectangle** within the image with random values (or the dataset mean, or zero) during training.

Three parameters:

- application probability (typically $0.25$),
- rectangle area / image area ratio (usually $[0.02, 0.4]$),
- aspect ratio (usually $[0.3, 3.3]$).

## Effect

- Forces the network to use **multiple regions of the image** rather than relying on a single cue (e.g., an animal's head).
- Improves robustness to **partial occlusion**, a natural form of distribution shift.
- Complementary to other spatial augmentations (RandAugment, MixUp).

## Why this reference for us

- Cited in `tech_notes.md::Augmentation Pipelines`.
- **Not enabled by default** in our current config (we stick to RandAugment alone for Stage B.2 to stay comparable to Sun 2020).
- **Possible extension**: test the effect of Random Erasing on robustness to CIFAR-10-C, which contains semi-local corruptions like `pixelate`, `jpeg_compression`.
