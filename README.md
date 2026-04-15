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
- **CIFAR-10-C** as the shifted-test benchmark for robustness and TTT analysis
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
- CIFAR-10-C for corrupted / shifted test-time evaluation

3. **Stage A: Self-Supervised Pretraining**
- Train ViT encoder with SimCLR objective (NT-Xent)
- Save pretrained encoder weights

4. **Stage B: Downstream Evaluation**
- Linear probe with frozen encoder as a comparison baseline
- Full supervised fine-tuning with unfrozen encoder as the main downstream stage

5. **Stage C: Evaluation Without TTT**
- Evaluate the trained classifier on clean CIFAR-10 test data
- Evaluate the same model on CIFAR-10-C to measure the effect of distribution shift

6. **Stage D: Test-Time Training**
- Adapt selected model parameters on clean and corrupted test batches without labels
- Evaluate predictions after a small number of adaptation steps

7. **Evaluation and Analysis**
- Top-1 accuracy on clean test set
- Accuracy under corruption / distribution shift
- Delta between without-TTT and with-TTT settings, especially on CIFAR-10-C
- Optional latency overhead analysis

## Repository Structure
```text
.
├── configs/                # Experiment configuration templates
├── docs/
│   ├── assignment/         # TER assignment PDFs
│   ├── notes/              # Technical notes on methods used
│   └── papers/             # Reference papers (SimCLR, TTT, ActMAD...)
├── notebooks/              # Colab-ready notebook(s) for running the project
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

## Technical Notes
The project notes used to explain the main design choices live in
[`docs/notes/tech_notes.md`](docs/notes/tech_notes.md). They summarize ViT,
SimCLR, linear probe vs fine-tuning, TTT, and why CIFAR-10-C matters for
distribution-shift evaluation.

## First Run
Start with a smoke test before launching a long training run.

### Local
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Edit `configs/default.yaml` and set `simclr.epochs: 1` for the first run.
4. Launch Stage A:
   ```bash
   python3 main.py --config configs/default.yaml
   ```
5. Check outputs:
   - logs: `logs/simclr-vit-cifar10-ter/experiment.log`
   - metrics: `logs/simclr-vit-cifar10-ter/metrics.csv`
   - checkpoints: `checkpoints/simclr-vit-cifar10-ter/simclr_best.pt`
   - encoder export: `checkpoints/simclr-vit-cifar10-ter/encoder_pretrained.pt`

### Google Colab
Use the notebook at
[`notebooks/run_stage_a_on_colab.ipynb`](notebooks/run_stage_a_on_colab.ipynb).

1. Download that notebook from the repository.
2. Open [Google Colab](https://colab.research.google.com/) and upload the notebook.
3. Switch the runtime to a GPU:
   - `Runtime -> Change runtime type -> T4 GPU` (or any available GPU)
4. Edit the first config cell in the notebook:
   - set `REPO_URL` to your repository URL
   - optionally set `BRANCH`
   - set `USE_GOOGLE_DRIVE = True` if you want logs and checkpoints persisted
5. Run the notebook cells from top to bottom.

The notebook will:
- clone the repository into Colab
- install `requirements.txt`
- show the active config
- run `python main.py --config configs/default.yaml`
- expose logs, metrics, checkpoints, and TensorBoard
- optionally copy artifacts to Google Drive

Keep the same smoke-test setting with `simclr.epochs: 1` for the first Colab run.

Use the 1-epoch smoke test to verify that config loading, dataset preparation,
logging, and checkpoint export all work. Only increase epochs after the smoke
test succeeds. If you do not have a local CUDA GPU, Colab is the recommended
way to run longer experiments.

## First Results

The first non-smoke Stage A run was completed with:
- dataset: CIFAR-10
- backbone: ViT tiny
- image size: 32
- patch size: 4
- projection dim: 128
- SimCLR epochs: 20
- optimizer: Adam with learning rate `0.001`

Observed SSL loss trend:

| Epoch | Train Loss | Val Loss |
|------:|-----------:|---------:|
| 1 | 4.82997 | 4.64490 |
| 10 | 4.26077 | 4.17320 |
| 20 | 4.13479 | 4.07346 |

Current interpretation:
- Stage A training is stable and improves throughout the 20 epochs.
- The best checkpoint was the final one at epoch 20 with `val/loss = 4.07346`.
- This is only a self-supervised pretraining signal, not a classification result yet.
- Downstream quality still needs to be measured with a linear probe and full fine-tuning.

Produced artifacts:
- pretrained encoder: `checkpoints/simclr-vit-cifar10-ter/encoder_pretrained.pt`
- best SimCLR checkpoint: `checkpoints/simclr-vit-cifar10-ter/simclr_best.pt`
- run log: `logs/.../simclr-vit-cifar10-ter/experiment.log`
- scalar history: `logs/.../simclr-vit-cifar10-ter/metrics.csv`

## Current Status

| Component | Status |
|-----------|--------|
| Data — CIFAR-10 splits, SimCLR transforms, DataLoaders | done |
| Models — ViT backbone (timm), SimCLR model, classifiers | done |
| Config — typed frozen dataclasses, YAML loader, section validation | done |
| Utils — `set_seed`, `ExperimentLogger` (CSV + TensorBoard), `CheckpointManager` | done |
| Base trainer — shared epoch loop, abstract train/validate hooks | done |
| Stage A — SimCLR pretraining (NT-Xent loss + trainer) | implemented, first 20-epoch run completed |
| Stage B — Linear probe + fine-tune trainers | not started |
| Stage C — TTT adapter | not started |
| Pipeline orchestration | implemented for Stage A |
| Evaluator / CIFAR-10-C integration | not started |
