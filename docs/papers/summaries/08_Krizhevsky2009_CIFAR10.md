# Krizhevsky 2009 — CIFAR-10 / CIFAR-100 (Tech Report)

**Reference**
Krizhevsky, A.
*Learning Multiple Layers of Features from Tiny Images.*
Tech. Report, University of Toronto, 2009.
[cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

Local PDF: [`CIFAR10_Krizhevsky2009.pdf`](../pdfs/CIFAR10_Krizhevsky2009.pdf)

---

## Core idea

Describes the construction of two reference datasets for low-resolution image classification:

- **CIFAR-10**: 60,000 color images of size $32 \times 32$, 10 balanced classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Standard split: 50,000 train / 10,000 test.
- **CIFAR-100**: 60,000 color images of size $32 \times 32$, 100 finer-grained classes, 600 images per class.

Images come from the **80 Million Tiny Images** dataset, manually filtered and annotated by the authors.

## Useful properties

- All images are already **centered** on the main object.
- No noise, no occlusion, controlled lighting.
- 10 natural, semantically distinct classes (no fine-grained breeds).

## Why this reference for us

- It is the **base dataset** of the project: Stage A pretrains SimCLR on CIFAR-10 without labels, Stage B fine-tunes with the 10 classes, Stage C evaluates on clean CIFAR-10 and on CIFAR-10-C (corrupted versions).
- **Choice of CIFAR-10 over ImageNet** is dictated by compute budget (L4 GPU, 200 SimCLR epochs ≈ 2 h vs >24 h on ImageNet) — detailed in our annex.

## Historical note

Although the report dates from 2009, CIFAR-10 remains a reference benchmark for fast methodological experiments — particularly in TTT, where the entire pipeline can be evaluated in a few hours.
