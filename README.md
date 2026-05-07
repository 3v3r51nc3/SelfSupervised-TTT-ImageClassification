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

5. **Stage C: Robustness + TTT Evaluation (combined)**
- Evaluate the fine-tuned classifier on clean CIFAR-10 and on CIFAR-10-C
  across 14 corruptions × 5 severities (the four families: noise, blur,
  weather, digital).
- For each (corruption, severity) compute both the **baseline** (no
  adaptation) and **TTT** (Sun 2020 TTT, rotation auxiliary head adapted
  per test image / batch) accuracy and loss in the same pass.
- The adapter is built once and reset between (corruption, severity)
  sets so adaptation never bleeds across them.
- Output: `logs/<exp>/cifar10c_results.csv` with per-row baseline /
  TTT / delta accuracy.

6. **Evaluation and Analysis**
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
│   ├── papers/             # Reference papers
│   │   ├── pdfs/           # PDF copies (SimCLR, TTT, ActMAD, ...)
│   │   └── summaries/      # One-page markdown summary per paper
│   └── presentation/       # Project status presentation (LaTeX/PDF)
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
[`docs/notes/tech_notes.md`](docs/notes/tech_notes.md). They cover:

- **Methods** — ViT, SimCLR, NT-Xent, linear probe vs fine-tune, TTT (high
  level), CIFAR-10 / CIFAR-10-C.
- **Training recipe glossary** — AMP, AdamW + weight decay, warmup +
  cosine scheduler factory.
- **Sun 2020 TTT (the actual TTT method)** — rotation auxiliary head,
  per-image / per-batch adaptation, snapshot/reset semantics across
  (corruption, severity).
- **Augmentation pipelines** — SimCLR vs supervised-train vs eval (table).
- **Stage orchestration & artifact flow** — how A → B.1 → B.2 → C feed
  each other.
- **Stage C CSV schema** — column-by-column breakdown of
  `cifar10c_results.csv`.
- **Regularization & stability knobs** — label smoothing, early stopping,
  gradient clipping under AMP, drop_path.
- **Config presets & notebook switch** — smoke vs default.
- **References** — full citation list (papers + arXiv links) for every
  technique used.

### Paper summaries

Per-paper one-page summaries of every reference cited in `tech_notes.md`
live in [`docs/papers/summaries/`](docs/papers/summaries/README.md). Each
file gives the citation, the core idea, key results, and how the paper
is used in this project. The full reading list (TTT methods, building
blocks, optimization, regularization, normalization layers) is indexed
in [`docs/papers/summaries/README.md`](docs/papers/summaries/README.md).
PDFs of all referenced papers are stored in `docs/papers/pdfs/`.

## First Run
Two configs are shipped:

- **`configs/smoke.yaml`** — 2 epochs per stage, ~5 min on a GPU. Use it
  to verify that the full A → B.1 → B.2 → C pipeline runs end-to-end and
  that `cifar10c_results.csv` is written.
- **`configs/default.yaml`** — overnight-grade run (SimCLR 200 ep,
  fine-tune 30 ep, linear probe 30 ep, full Stage C eval) with AMP,
  AdamW, warmup + cosine, RandAugment, label smoothing, drop_path,
  early stopping, and Sun 2020 TTT enabled.

Always run smoke first, then switch to default for the real run.

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
3. Smoke test:
   ```bash
   python3 main.py --config configs/smoke.yaml
   ```
4. Overnight run (only after smoke succeeds):
   ```bash
   python3 main.py --config configs/default.yaml
   ```
5. Check outputs (replace `<exp>` with `experiment.name` from the YAML):
   - logs: `logs/<exp>/experiment.log`
   - metrics: `logs/<exp>/metrics.csv`
   - SimCLR checkpoint: `checkpoints/<exp>/simclr_best.pt`
   - encoder export: `checkpoints/<exp>/encoder_pretrained.pt`
   - linear probe: `checkpoints/<exp>/linear_probe_best.pt`
   - fine-tune: `checkpoints/<exp>/finetune_best.pt`
   - Stage C report: `logs/<exp>/cifar10c_results.csv`

### Google Colab
A single evergreen notebook drives every stage:
[`notebooks/colab.ipynb`](notebooks/colab.ipynb).

1. Download `notebooks/colab.ipynb` and upload it to
   [Google Colab](https://colab.research.google.com/).
2. Switch the runtime to a GPU:
   - `Runtime -> Change runtime type -> T4 GPU` (or any available GPU).
3. Edit the **Configuration** cell:
   - set `REPO_URL` and `BRANCH`,
   - choose `CONFIG_PRESET = "smoke"` for a 5-minute pipeline check or
     `"default"` for the overnight run; the `CONFIG_PATHS` dict maps the
     preset to the YAML on disk,
   - optionally patch individual fields via `CONFIG_OVERRIDES` without
     editing YAML,
   - leave `USE_GOOGLE_DRIVE = True` to persist logs and checkpoints,
   - flip the `RUN_STAGE_A`, `RUN_STAGE_B1`, `RUN_STAGE_B2`, `RUN_STAGE_C`
     booleans to choose which stages run.
4. Run the notebook cells from top to bottom.

The notebook will:
- clone the repository into Colab and install `requirements.txt`,
- restore prior `logs/` and `checkpoints/` from Google Drive (so a skipped
  upstream stage can reuse the artifact it produced last time),
- download CIFAR-10-C automatically when Stage C is enabled,
- run only the selected stages via `ExperimentPipeline.run_stage_*`,
- expose logs, metrics, a CIFAR-10-C summary table, and TensorBoard,
- copy artifacts back to Google Drive.

Stage dependencies (when a stage is skipped, the upstream artifact must
already be on disk - the restore-from-Drive step handles this):
- Stage B.1 / B.2 require Stage A's `encoder_pretrained.pt`.
- Stage C requires Stage B.2's `finetune_best.pt`.

If you do not have a local CUDA GPU, Colab is the recommended way to run
longer experiments. The overnight `default` config targets an L4 / A100
(roughly 2–3 h on an L4); plain T4 is workable but slower.

## Preliminary Results (first overnight run, 2026-04-30)

First end-to-end run with the overnight recipe (SimCLR 200 ep, fine-tune
30 ep, full Stage C eval over 14 corruptions × 5 severities). Logs in
[`logs/simclr-vit-cifar10-ter/`](logs/simclr-vit-cifar10-ter/).

### Clean accuracy
| Setting | Top-1 |
|---|---|
| Stage B.2 fine-tune val (epoch 30, best) | **0.9008** |
| Stage C clean baseline (no TTT) | 0.8959 |
| Stage C clean + per-batch TTT | 0.8960 (Δ +0.0001) |
| Stage C clean + per-image TTT (1k subset) | 0.8980 |

No degradation on clean data — the rotation auxiliary head does not
break the classifier when adapted at test time.

### CIFAR-10-C (mean accuracy over 14 corruptions)
TTT params: `lr=1e-3`, `steps=1`, `lambda_rot=1.0`,
`adapt_scope=encoder_plus_head`, `rotation_mode=rand`.

| Severity | baseline | per-batch | per-image (1k) | Δ batch | Δ image* |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8673 | 0.8679 | 0.8666 | +0.0007 | -0.0006 |
| 2 | 0.8392 | 0.8409 | 0.8395 | +0.0017 | +0.0003 |
| 3 | 0.8033 | 0.8059 | 0.8033 | +0.0025 |  0.0000 |
| 4 | 0.7516 | 0.7575 | 0.7534 | +0.0059 | +0.0018 |
| 5 | 0.6803 | 0.6890 | 0.6830 | +0.0087 | +0.0027 |
| **mean** | **0.7883** | **0.7922** | **0.7892** | **+0.0039** | **+0.0009** |

\* per-image deltas above are vs the full 10 k baseline; on the matched
1 k subset (apples-to-apples) per-image actually averages **Δ = −0.0048**
with **16 wins / 46 losses / 8 ties** out of 70 cells.

### Win/loss per cell (70 cells, severity ≥ 1)
| Mode | wins | losses | ties |
|---|---:|---:|---:|
| per-batch (vs full baseline) | **61** | 6 | 3 |
| per-image (vs subset baseline) | 16 | **46** | 8 |

### Headline corruptions (severity 5, biggest gains for per-batch)
| Corruption | baseline | per-batch | Δ |
|---|---:|---:|---:|
| fog | 0.5367 | 0.5586 | **+0.0219** |
| pixelate | 0.7378 | 0.7530 | +0.0152 |
| impulse_noise | 0.3959 | 0.4109 | +0.0150 |
| pixelate s4 | 0.7634 | 0.7772 | +0.0138 |
| contrast | 0.5919 | 0.6039 | +0.0120 |
| gaussian_noise s4 | 0.5556 | 0.5679 | +0.0123 |

### Interpretation
1. **Per-batch TTT works as advertised**: small but consistent gain that
   grows with severity (+0.07 pp at sev 1 → +0.87 pp at sev 5), 61/70
   cells improved, no regression on clean data. This is the expected
   Sun 2020 TTT signature.
2. **Per-image TTT degrades on this ViT/LN setup**. Adapting one image
   at a time produces a net negative Δ on a matched subset, losing on
   46/70 cells. Hypotheses (ranked):
   - ViT uses LayerNorm, so per-image adaptation cannot exploit the
     batch-statistics regularization that BN-based TTT relies on;
   - 128 rotated copies of a single image is a thin signal — the
     `lr=1e-3, steps=1` recipe was tuned for batches, not single
     images;
   - encoder + rotation-head scope may be over-parameterized for a
     single-image update, causing the encoder to drift.
3. **Gains concentrate on heavy corruptions** (fog 5, pixelate 4-5,
   noise 4-5, contrast 5). These are the regimes where the rotation
   pretext task has the most signal-to-noise to give back.

This is a **preliminary** result on a single seed — variance bars and
the lr/steps ablation are needed before claiming the per-image
degradation is a property of the method rather than the recipe.

## Next Steps
1. **Ablate the per-image recipe** using the configs already in
   [`configs/ablation/`](configs/ablation/): `lr_1e-4.yaml`,
   `lr_1e-5.yaml`, `steps_5.yaml`, `steps_10.yaml`,
   `lr_1e-5_steps_10.yaml`, `expand_k1.yaml`. Check if a smaller LR or
   more steps reverse the per-image regression.
2. **Multi-seed re-run** of the default config (≥3 seeds) to put error
   bars on the +0.39 pp per-batch mean gain — the gain is small enough
   that single-seed noise could dominate.
3. **Restrict adapt scope** to `encoder` only or to `head` only, to
   isolate where the per-image drift originates.
4. **Aggregate plots**: severity curves and per-corruption bar charts
   from `cifar10c_results.csv` for the report (notebooks/colab.ipynb
   already loads the long-format summary).
5. **Latency bookkeeping**: log per-cell adaptation time so the
   per-image vs per-batch trade-off is reported in both accuracy and
   wall-clock.

## Current Status

| Component | Status |
|-----------|--------|
| Data — CIFAR-10 splits, three transform pipelines (SimCLR / sup train / eval), DataLoaders | done |
| Models — ViT backbone (timm) with `drop_path_rate`, SimCLR projector, classifiers | done |
| Config — typed frozen dataclasses, YAML loader, section validation, `smoke` + `default` presets | done |
| Utils — `set_seed`, `ExperimentLogger` (CSV + TensorBoard), `CheckpointManager` with `reset_best()` | done |
| Base trainer — shared epoch loop, AMP (`autocast` + `GradScaler`), grad clipping, early stopping | done |
| Stage A — SimCLR pretraining (NT-Xent + AdamW + warmup-cosine + AMP) | done |
| Stage B.1 — Linear probe (frozen encoder, eval transforms) | done |
| Stage B.2 — Fine-tune (label smoothing, RandAugment, drop_path, early stopping) | done |
| Stage C — CIFAR-10-C eval over all corruptions × severities, baseline + TTT in one pass, CSV report | done |
| TTT adapter — Sun 2020 TTT (rotation auxiliary, snapshot/reset per cell) | done |
| Stage B.2 trainer — joint CE + rotation loss with `lambda_rot` | done |
| Pipeline orchestration — A → B.1 → B.2 → C with `reset_best()` between stages | done |
| Evaluator — `evaluate` and `evaluate_with_ttt` | done |
| Notebook — `CONFIG_PRESET` switch, Drive caching, CIFAR-10-C auto-download | done |
| Overnight run on L4/A100 with `configs/default.yaml` | done (2026-04-30, single seed) |
| TTT lr/steps ablation sweep | pending |
| Multi-seed re-run for variance bars | pending |

## References

Detailed citations and discussion live in
[`docs/notes/tech_notes.md`](docs/notes/tech_notes.md). Short list:

### Architecture
- **ViT** — Dosovitskiy, A. *et al.* "An Image is Worth 16x16 Words:
  Transformers for Image Recognition at Scale." *ICLR 2021.*
  [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- **timm library** — Wightman, R. "PyTorch Image Models." 2019.
  [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)

### Self-supervised pretraining
- **SimCLR / NT-Xent** — Chen, T., Kornblith, S., Norouzi, M., Hinton, G.
  "A Simple Framework for Contrastive Learning of Visual Representations."
  *ICML 2020.* [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)

### Test-time adaptation
- **TTT (Sun 2020, the method actually implemented)** — Sun, Y.,
  Wang, X., Liu, Z., Miller, J., Efros, A., Hardt, M. "Test-Time
  Training with Self-Supervision for Generalization under Distribution
  Shifts." *ICML 2020.*
  [arXiv:1909.13231](https://arxiv.org/abs/1909.13231).
  Reference implementation:
  [github.com/yueatsprograms/ttt_cifar_release](https://github.com/yueatsprograms/ttt_cifar_release).

### Datasets
- **CIFAR-10** — Krizhevsky, A. "Learning Multiple Layers of Features
  from Tiny Images." Tech. Report, 2009.
  [cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- **CIFAR-10-C** — Hendrycks, D., Dietterich, T. "Benchmarking Neural
  Network Robustness to Common Corruptions and Perturbations."
  *ICLR 2019.* [arXiv:1903.12261](https://arxiv.org/abs/1903.12261)

### Optimization & scheduling
- **AdamW** — Loshchilov, I., Hutter, F. "Decoupled Weight Decay
  Regularization." *ICLR 2019.*
  [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- **Cosine annealing (SGDR)** — Loshchilov, I., Hutter, F. "SGDR:
  Stochastic Gradient Descent with Warm Restarts." *ICLR 2017.*
  [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
- **Linear LR warmup (large-batch training)** — Goyal, P. *et al.*
  "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." 2017.
  [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
- **Mixed precision training** — Micikevicius, P. *et al.* "Mixed
  Precision Training." *ICLR 2018.*
  [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)

### Regularization
- **Label smoothing** — Szegedy, C. *et al.* "Rethinking the Inception
  Architecture for Computer Vision." *CVPR 2016.*
  [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
- **Stochastic depth (drop_path)** — Huang, G. *et al.* "Deep Networks
  with Stochastic Depth." *ECCV 2016.*
  [arXiv:1603.09382](https://arxiv.org/abs/1603.09382)
- **RandAugment** — Cubuk, E. D., Zoph, B., Shlens, J., Le, Q. V.
  "RandAugment: Practical Automated Data Augmentation with a Reduced
  Search Space." *NeurIPS 2020.*
  [arXiv:1909.13719](https://arxiv.org/abs/1909.13719)
- **Random Erasing** — Zhong, Z. *et al.* "Random Erasing Data
  Augmentation." *AAAI 2020.*
  [arXiv:1708.04896](https://arxiv.org/abs/1708.04896)

### Normalization
- **LayerNorm** — Ba, J. L., Kiros, J. R., Hinton, G. E. "Layer
  Normalization." 2016.
  [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- **BatchNorm** — Ioffe, S., Szegedy, C. "Batch Normalization." *ICML 2015.*
  [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- **GroupNorm** — Wu, Y., He, K. "Group Normalization." *ECCV 2018.*
  [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)

### Frameworks
- **PyTorch** — Paszke, A. *et al.* "PyTorch: An Imperative Style,
  High-Performance Deep Learning Library." *NeurIPS 2019.*
  [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)
