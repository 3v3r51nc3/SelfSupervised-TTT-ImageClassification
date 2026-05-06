# Hendrycks & Dietterich 2019 — CIFAR-10-C / Robustness Benchmark

**Reference**
Hendrycks, D., Dietterich, T.
*Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.*
ICLR 2019. [arXiv:1903.12261](https://arxiv.org/abs/1903.12261).

Local PDF: [`CIFAR10C_Hendrycks2019_1903.12261.pdf`](../pdfs/CIFAR10C_Hendrycks2019_1903.12261.pdf)

---

## Core idea

Builds a standardized **robustness benchmark** by applying a fixed set of corruptions to the CIFAR-10 (and ImageNet, MNIST) test set. Allows measuring how a classifier trained on clean data degrades under distribution shift.

## Benchmark structure

- **15 "test" corruptions** in 4 families:
  - **Noise**: gaussian, shot, impulse
  - **Blur**: defocus, glass, motion, zoom
  - **Weather**: snow, frost, fog, brightness
  - **Digital**: contrast, elastic, pixelate, jpeg
- **5 severity levels** (1 = mild, 5 = very degraded) per corruption.
- **4 "extra" corruptions** held out for hyperparameter tuning: `speckle_noise`, `gaussian_blur`, `spatter`, `saturate`.

Total: 15 × 5 = **75 cells** per model for the standard benchmark, plus 4 × 5 extra cells.

## Metric: mCE (mean Corruption Error)

Mean error across the 15 × 5 cells, normalized by a reference model (often AlexNet) to compare across architectures.

## Why this reference for us

- This is **the benchmark on which we evaluate Stage C**. See `src/data/dataset.py::cifar10c_corruptions()`.
- Generates the results table `cifar10c_results.csv`.
- **Important note**: our code covers 14/15 corruptions (`zoom_blur` is missing) — see Annexe A in the presentation.

## Connection to TTT evaluation

This is the de-facto standard for evaluating any TTT method on images. Sun 2020, Liu 2021 (TTT++), Wang 2021 (TENT), Mirza 2023 (ActMAD) all report their results on this benchmark — which makes it possible to compare our setup with the existing landscape.
