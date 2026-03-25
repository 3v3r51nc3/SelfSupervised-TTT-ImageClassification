# SelfSupervised-TTT-ImageClassification

## TER Context
This repository is the TER project for the second semester of **M1 VMI**.  
The project focuses on self-supervised learning and test-time adaptation for image classification.

## Project Team and Supervision
**Students**
- Maksym DOLHOV
- Fallou DIOUF

**Supervisors**
- Ayoub Karine (ayoub.karine@u-paris.fr)
- Camille Kurtz (camille.kurtz@u-paris.fr)
- Laurent Wendling (Laurent.Wendling@u-paris.fr)

## Problem Statement
Standard image classifiers often lose performance when test data distribution shifts from training data.
This is a key issue for real-world deployment, where corruption, noise, and domain shift are common.

The project investigates whether a self-supervised representation learning strategy can improve robustness,
and whether **Test-Time Training (TTT)** can recover performance at inference time.

## Main Goal
Build and evaluate a complete pipeline based on:
- **SimCLR** for self-supervised pretraining
- **ViT** as the encoder backbone
- **CIFAR-10** as the primary small-scale benchmark
- **TTT** for online adaptation during inference

## Pipeline Plan
1. **Configuration and Reproducibility**
- Central experiment config
- Fixed random seeds
- Logging and checkpoints

2. **Data Preparation**
- CIFAR-10 train/validation/test splits
- SimCLR augmentations (two strong views per image)
- Evaluation transforms
- Optional CIFAR-10-C for corruption robustness analysis

3. **Stage A: Self-Supervised Pretraining**
- Train ViT encoder with SimCLR objective (NT-Xent)
- Save pretrained encoder weights

4. **Stage B: Downstream Evaluation**
- Linear probe with frozen encoder
- Full supervised fine-tuning with unfrozen encoder

5. **Stage C: Test-Time Training**
- Adapt selected model parameters on test batches
- Evaluate predictions after a small number of adaptation steps

6. **Evaluation and Analysis**
- Top-1 accuracy on clean test set
- Accuracy under corruption/domain shift
- Delta between without-TTT and with-TTT settings
- Optional latency overhead analysis

## Repository Structure
```text
.
├── configs/                # Experiment configuration templates
├── docs/
│   ├── assignment/         # TER assignment PDFs
│   ├── notes/              # Technical notes on methods used
│   └── papers/             # Reference papers (SimCLR, TTT, ActMAD...)
├── main.py                 # Entry point and pipeline overview
└── src/
    ├── core/               # Config + pipeline orchestration
    ├── data/               # Datasets and transforms
    ├── models/             # ViT backbone, SimCLR model, classifier
    ├── training/           # Pretrain/probe/finetune trainers
    ├── ttt/                # Test-time adaptation components
    ├── evaluation/         # Metrics and evaluation routines
    └── utils/              # Logging and checkpoint helpers
```

## Current Status

| Stage | Status |
|-------|--------|
| Data — CIFAR-10 splits, SimCLR transforms, DataLoaders | done |
| Models — ViT backbone (timm), SimCLR model, classifier | done |
| Config + pipeline orchestration | in progress |
| Stage A — SimCLR pretraining (NT-Xent) | not started |
| Stage B — Linear probe + fine-tune | not started |
| Stage C — TTT adapter | not started |
| Evaluation + analysis | not started |
