# Technical Notes

## ViT (Vision Transformer)

Standard image classifiers (CNNs like ResNet) use convolutions to scan the image.
ViT takes a different approach — it splits the image into fixed-size patches and
treats them like a sequence of tokens, the same way a text Transformer handles words.

For CIFAR-10 (32×32 images) with patch_size=4:
- we get an 8×8 grid = 64 patches
- each patch is flattened into a vector and projected to embed_dim=192
- a standard Transformer encoder then processes the sequence
- the output is a single 192-dim vector = the image embedding

We use ViT-Tiny (smallest variant) because CIFAR-10 is small and we don't have
a lot of compute. Larger variants (Small, Base) would overfit or train too slowly.

---

## SimCLR

SimCLR is a self-supervised learning method — it learns visual representations
without any labels. The core idea: an image and its augmented version should
produce similar embeddings, while two different images should be far apart.

Training pipeline for one batch:
1. Take N images
2. Apply two random augmentations to each → 2N views total
3. Pass all 2N views through encoder + projector → 2N embedding vectors
4. Compute NT-Xent loss: pull positive pairs (same image) together,
   push all other pairs (different images) apart

After training we split the model in two parts:
- encoder = the ViT that takes a 32×32 image and outputs a 192-dim vector.
  This is the part that actually learned to understand images. We keep it.
- projector = the small 2-layer network on top that maps 192 → 128 dims.
  It only existed to make NT-Xent work during pretraining. We throw it away.

We then attach a new linear layer (192 → 10 classes) on top of the encoder
and train it to classify CIFAR-10. The encoder never saw a single label
during pretraining — it learned purely from comparing pairs of augmented images.

---

## NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)

This is the loss function of SimCLR. For a positive pair (i, j):

```
similarity(i, j) = cosine_similarity(z_i, z_j) / temperature

loss(i, j) = -log(
    exp(similarity(i, j)) /
    sum over all k≠i [ exp(similarity(i, k)) ]
)
```

The final loss is averaged over all 2N positive pairs in the batch.

Key parameters:
- temperature τ = 0.2 (from config) — lower = sharper distribution,
  model is forced to be more precise about which pairs are similar
- batch size = 1024 → 2046 negatives per pair (more negatives = better signal)
  this is why drop_last=True is needed — a smaller batch gives fewer negatives
  and makes the loss less stable

---

## Linear Probe vs Fine-Tune

After SimCLR pretraining we want to test if the learned embeddings are useful
for classification. Two ways to do this:

**Linear Probe**
- freeze the encoder completely (no gradient flows through it)
- train only a single linear layer on top: embedding → 10 classes
- if accuracy is high, the encoder learned good representations
- fast to train, used as a benchmark

**Fine-Tune**
- unfreeze the encoder, train everything end-to-end
- usually gives better accuracy than linear probe
- but requires more compute and risks forgetting the SSL representations

For this TER, **fine-tuning is the main downstream objective** because the
assignment explicitly asks for a self-supervised pretrained model that is then
fine-tuned on an image classification task. The **linear probe remains useful**
as a comparison baseline: it tells us how strong the SSL representations already
are before full supervised adaptation.

---

## TTT (Test-Time Training)

Standard models are trained once and then frozen during inference.
The problem: if test data looks different from training data (different lighting,
noise, corruption), accuracy drops significantly.

TTT fixes this by doing a few gradient steps on the test stream before
predicting, using a **self-supervised** objective so no test labels are
needed:

1. Take a test sample (or batch). No labels.
2. Apply an SSL objective for which a synthetic label can be generated
   from the input itself.
3. Take a small number of SGD steps on a subset of the model.
4. Predict the class with the updated model.
5. Reset model weights so adaptation does not bleed across cells.

The concrete instantiation used in this project is **Sun 2020 TTT
(rotation auxiliary head)** — the SSL objective is "predict by what
angle (0/90/180/270°) this input was rotated", and the small subset of
the model that gets adapted is `encoder + rotation_head` while the
classifier head stays frozen. Full details: see the *Sun 2020 TTT*
section below.

For the TER objective, TTT must be evaluated **without access to test
labels** and compared against the same model **without TTT**. The key
question is not just whether the model predicts well, but whether this
test-time adaptation actually helps under distribution shift.

### Self-supervised vs unsupervised — the reason TENT was rejected

Both terms describe label-free training, but they are not the same
thing. TER2.pdf explicitly asks for **auto-supervisé** (self-supervised)
test-time adaptation, which rules out one whole family of methods.

| | Unsupervised | Self-supervised |
|---|---|---|
| Has a prediction target? | No — the objective looks at the data alone (clusters, density, low-rank structure). | Yes — a synthetic target is **generated from the input itself** (rotation angle, masked pixel, contrastive pair). |
| Loss form | likelihood, reconstruction, entropy of own predictions, … | standard supervised loss (usually CE) against the synthetic label. |
| Examples | K-means, PCA, autoencoder reconstruction, **TENT** (entropy minimization on the model's own logits). | SimCLR (contrastive pair labels), MAE (masked-pixel target), **Sun 2020 TTT** (rotation-angle label). |

**TENT** (Wang *et al.* ICLR 2021) adapts at test time by minimizing
the entropy of the classifier's own softmax distribution — it never
constructs a synthetic prediction target, just sharpens the existing
one. By the column above this is *unsupervised*, not self-supervised,
which is why it does not satisfy "auto-supervisé" in the TER and was
removed from this project.

**Sun 2020 TTT** is squarely in the self-supervised column: the
rotation auxiliary takes the test image, applies one of four rotations,
*and the rotation index becomes the supervised label* for a CE loss on
`rotation_head`. The label is fabricated from the input via a
deterministic transform — exactly the SimCLR / MAE pattern, just with
"angle of rotation" as the pretext task.

This is also why the rotation auxiliary is wired in at training time
(joint CE + rotation in Stage B.2): the rotation head needs to be
trained on clean data first so that at test time its gradient signal
is meaningful. TENT does not have this two-stage shape because it
never trained an auxiliary head.

---

## CIFAR-10 and CIFAR-10-C

60 000 images total, 10 classes (airplane, car, bird, cat, deer,
dog, frog, horse, ship, truck), 32×32 pixels, RGB.

Split used in this project:
- 45 000 train (SSL pretraining + supervised fine-tune)
- 5 000 validation (hyperparameter tuning)
- 10 000 test (final evaluation)

CIFAR-10-C is the same test set but with 14 types of corruption
(four families: noise, blur, weather, digital) at 5 severity levels. In this project it is not
just a side benchmark: it is the main way to evaluate how much TTT helps under
distribution shift, because the assignment focuses on adaptation to shifted test
conditions.

---

## How to Interpret the Experiments

The final experimental pipeline should answer four slightly different questions:

**1. Linear Probe**
- If we freeze the SSL encoder and train only a linear head, do the learned
  representations already separate CIFAR-10 classes well?

**2. Fine-Tune**
- If we unfreeze the encoder and train end-to-end on labeled data, what is the
  main downstream classification performance of the pretrained model?

**3. Without TTT**
- How much performance do we lose when the fine-tuned model is evaluated on
  shifted data such as CIFAR-10-C without any adaptation at test time?

**4. With TTT**
- Can a few self-supervised adaptation steps at test time recover some of that
  lost performance, especially on corrupted or shifted test inputs?

In practice, this gives a clean progression:
- linear probe = representation quality baseline
- fine-tune = main supervised downstream result
- no TTT = test-time baseline
- with TTT = adaptation result

This matches the TER assignment well:
- pretrain a model in self-supervision
- fine-tune it on classification
- adapt it at test time without labels
- compare with and without adaptation under distribution shift

---

## Training Recipe Glossary (AMP, AdamW, Scheduler)

These three ingredients are the standard "transformer training recipe" of the
last 3–4 years and are all wired into this project.

### AMP (Automatic Mixed Precision)

Mixed-precision training: forward and backward passes run in float16 (or
bfloat16), while master weights and gradient accumulations stay in float32.
On L4 / A100 this gives a 1.5–2× speedup and roughly 50% VRAM savings, almost
for free.

In PyTorch this is two pieces working together:
- `torch.amp.autocast()` — a context manager that runs the wrapped operations
  in float16.
- `GradScaler` — scales the loss before `.backward()` so that very small
  gradients do not underflow to zero, then unscales them before the optimizer
  step.

In this project both are wired into `BaseTrainer._backward_step` and
`BaseTrainer._autocast`, and enabled per-stage via the `use_amp` flag in the
config (SimCLR / linear probe / fine-tune).

### AdamW + Weight Decay

- **Adam** — standard adaptive optimizer with first- and second-order moment
  estimates of the gradient.
- **AdamW** — fix from Loshchilov & Hutter (2017): weight decay (L2-style
  regularization on the weights) is decoupled from the gradient update,
  instead of being baked into it. Vanilla Adam handles weight decay poorly
  because the adaptive learning rate "consumes" it.
- **Weight decay (wd)** — at every step the weights are multiplied by
  `(1 - lr · wd)`, so they are slowly pulled toward zero. The bigger the wd,
  the stronger the penalty on large weights and the less the model overfits.

How wd is set in this project:
- `wd = 0.05` for fine-tune — the standard value for ViT models.
- `wd = 1e-4` for SimCLR — light decay during SSL.
- `wd = 0` for the linear probe — a single linear layer needs no extra
  regularization on top of frozen features.

### Scheduler factory (warmup + cosine)

Inside `pipeline.py`, the helper `_build_warmup_cosine()` assembles the
learning-rate schedule used by every training stage:

1. **Linear warmup** for the first `warmup_epochs` — LR rises from
   approximately zero to the base LR. This matters because at the very start
   of training the weights are random and a large LR would immediately blow
   them up. It is especially important for ViT models and for large batch
   sizes.
2. **Cosine annealing** after warmup — LR smoothly decays from the base value
   down to `eta_min` (1% of the base) following a cosine curve. This gives
   noticeably better convergence than a constant or step-wise LR.

Internally this is a `SequentialLR([LinearLR, CosineAnnealingLR])` from
`torch.optim.lr_scheduler`. It is called a "factory" because a single helper
assembles the right combination for any stage — SimCLR, fine-tune, or linear
probe — by parametrizing it on `total_epochs` and `warmup_epochs`.

### Why all three together

- AMP makes training fast and frees up VRAM for larger batches.
- AdamW + weight decay fights overfitting in the supervised stage.
- Warmup + cosine gives a stable start and a deeper, smoother convergence.

Together they let SimCLR pretraining go to 200 epochs on an L4 in roughly
two hours and let fine-tuning generalize instead of memorizing the 45k
labeled training images.

---

## Sun 2020 TTT — How TTT Is Actually Implemented

The concrete adaptation method used in this project is **Sun 2020 TTT
(rotation auxiliary head)** from Sun *et al.* ICML 2020. It matches the
TER2.pdf assignment ("Test-Time Training **auto-supervisé**" / "adapté
**individuellement à chaque image** de test") because (a) the loss at
test time is rotation-prediction CE — fully self-supervised — and (b) the
adapter snapshots and restores model state per evaluation cell, which
trivially extends to per-image adaptation via `rotation_mode = "expand"`.

### Y-shape model (`src/models/classifier.py::TTTModel`)

A shared `encoder` (ViT-tiny) feeds two heads:

- `classifier`     — `nn.Linear(192, 10)`, used at inference.
- `rotation_head`  — `nn.Linear(192, 4)`, SSL auxiliary, predicts the
  rotation angle (0/90/180/270) applied to the input.

`forward(x)` returns class logits; `forward_rotation(x)` returns rotation
logits. Both share the encoder, so adapting the encoder at test time
moves the classifier as well (the classifier itself stays frozen — see
adapter section).

### Stage B.2 — joint training (`src/training/ttt_finetune_trainer.py::TTTFineTuneTrainer`)

Loss per batch:

```
L = CE(classifier(x), y, label_smoothing=0.1)
  + λ_rot · CE(rotation_head(rotate(x, "rand")), rot_labels)
```

`λ_rot = 1.0` (Sun 2020 default, configurable via `ttt.lambda_rot`).
Validation reports only the supervised CE / top-1 on un-rotated inputs,
so the numbers stay directly comparable to a plain fine-tune baseline.

### Stage C — adapter (`src/ttt/adapter.py::TestTimeAdapter`)

The adapter exposes two adaptation modes that share the same
snapshot/reset machinery and the same SGD optimizer over `encoder + rotation_head`.

**Per-batch (`adapt_and_predict(images)`)** — operational baseline.

1. Set `model.train()` everywhere except `model.classifier`, which goes
   to `eval()` and has `requires_grad=False`. The supervised head must
   not be perturbed at test time.
2. For `K = steps` iterations (default 1, the Sun 2020 `niter`):
   - `rotated, rot_labels = rotate_batch(images, rotation_mode)`
   - `rot_logits = model.forward_rotation(rotated)`
   - `loss = CE(rot_logits, rot_labels)`
   - `optimizer.zero_grad(); loss.backward(); optimizer.step()`
3. With `no_grad`, return `model(images)` classification logits.

**Per-image (`adapt_and_predict_per_image(image, k)`)** — TER-strict.

1. Replicate the single test image into a batch of `k` copies
   (`per_image_aug_k`, default 128 — Sun 2020 default).
2. Apply `mode="rand"` rotations across the K-batch and run `steps`
   SGD steps exactly as above.
3. Forward the original (un-rotated) image through `classifier` under
   `no_grad`; argmax.
4. The caller must `reset()` between images so adaptation never leaks
   across samples.

The optimizer is **SGD** over `encoder.parameters() + rotation_head.parameters()`
with `lr = 1e-3` (Sun 2020 default). `rotation_mode` is `"rand"` for
random per-sample labels (default) or `"expand"` for the variant that
quadruples the input with all four fixed rotation labels.

### Snapshot / reset semantics

Test-time adaptation must not bleed across (corruption, severity)
combinations — Gaussian-blur severity 5 should not warm-start the
adapter for the next corruption. The adapter handles this by:

1. Capturing `_initial_model_state = deepcopy(model.state_dict())` and
   `_initial_optim_state = deepcopy(optimizer.state_dict())` in
   `__init__`. The snapshot reflects the **clean joint-trained weights**.
2. Exposing `reset()` which loads both snapshots back. The pipeline
   calls `adapter.reset()` once at the start of each (corruption, severity).

The pipeline builds the adapter exactly **once** before the corruption
loop. If we built a new adapter per corruption, each new snapshot would
capture weights that had already been mutated by the previous one.

### Per-batch vs per-image — wall-clock & TER strategy

The choice of mode is purely a Stage C concern (training is identical).
On A100/L4 at the Sun 2020 default of K=128 augmented copies + 1 SGD
step per image, the cost is:

| Mode | Per-image cost | Cells | Wall-clock (L4) |
|---|---|---|---|
| per-batch (full 10 000 / cell) | ~1 ms | 71 | ~5 min |
| per-image (full 10 000 / cell)  | ~100 ms | 71 | ~20 h — prohibitive |
| per-image (subsample 1 000 / cell) | ~100 ms | 71 | ~2 h — acceptable |

The TER assignment ("adapté **individuellement à chaque image** de
test") asks for per-image. To keep the pipeline runnable in one
overnight Colab session **and** report the TER-faithful headline, the
project runs both modes in the same Stage C session and dumps both
into the same CSV:

1. **per-batch** on the full 10 000-image test set per cell — fast
   ablation baseline.
2. **per-image** on a stratified random subsample of `per_image_subsample`
   images per cell (default 1 000, fixed seed) — TER-strict headline.
   At n = 1 000 the 95% Wilson CI for accuracy is approximately ±1.0 pp,
   below the granularity of the comparisons drawn in the report.

Both modes use the identical adapter — same K=128 augmentation count,
same SGD optimizer, same snapshot/reset protocol — so the only thing
that varies between rows in the CSV is whether the SGD step adapted on
one batch of distinct test inputs (per-batch) or on K copies of one
test input (per-image).

### Reference

Adapted from `yueatsprograms/ttt_cifar_release` —
`utils/rotation.py` (the rotation utilities) and
`test_calls/test_adapt.py::adapt_single` (the per-image adapt loop).

---

## Augmentation Pipelines

Three different transform pipelines run in this project, and they are not
interchangeable. They live in `src/data/transforms.py`:

| Pipeline | Used for | Transforms |
|---|---|---|
| **SimCLR** (`SimCLRTransform`) | SSL pretraining (Stage A) | `RandomResizedCrop(32, scale=(0.2, 1.0))` → `RandomHorizontalFlip` → `ColorJitter(0.4, 0.4, 0.4, 0.1)` p=0.8 → `RandomGrayscale` p=0.2 → `GaussianBlur` (kernel=3) → `ToTensor` → `Normalize` |
| **Supervised train** (`SupervisedTrainTransform`) | Stage B.2 fine-tune train split | `RandomCrop(32, padding=4, padding_mode='reflect')` → `RandomHorizontalFlip` → `RandAugment(n=2, m=9)` → `ToTensor` → `Normalize` → `RandomErasing(p=0.25)` |
| **Eval** (`EvalTransform`) | Validation, test, linear probe, all CIFAR-10-C | `ToTensor` → `Normalize` only |

The supervised-train pipeline is the **anti-overfitting workhorse** for
fine-tuning. Without it, val loss climbs after epoch 5 even on a properly
pretrained encoder. RandAugment adds 2 random ops at magnitude 9 per image,
RandomErasing zeroes a rectangular patch with probability 0.25, and the
reflect-padded random crop adds spatial jitter.

The linear probe deliberately uses **eval transforms** (no augmentation):
the encoder is frozen, so we want the probe to measure representation
quality, not augmentation quality.

For SimCLR, two **independent** samples from `SimCLRTransform` form the
positive pair (`view_a`, `view_b`). The dataset wrapper that produces
those pairs is in `src/data/dataset.py`.

---

## Stage Orchestration & Artifact Flow

The pipeline (`src/core/pipeline.py`) runs four stages in strict order:

1. **Stage A — SimCLR pretraining**
   - Inputs: unlabeled CIFAR-10 train split (45k images).
   - Output artifact: `checkpoints/<exp>/encoder_pretrained.pt`
     (encoder state_dict only — projector is dropped).
   - Optimizer: AdamW, scheduler: warmup + cosine, AMP on.

2. **Stage B.1 — Linear probe**
   - Inputs: `encoder_pretrained.pt`, frozen.
   - Trains: a single `nn.Linear(192, 10)` classifier head.
   - Output artifact: `checkpoints/<exp>/linear_probe_best.pt`.
   - Reports clean test accuracy as the *representation-quality baseline*.

3. **Stage B.2 — Fine-tune**
   - Inputs: `encoder_pretrained.pt`, **un**frozen.
   - Trains: encoder + classifier end-to-end with full supervised pipeline
     (label smoothing, RandAugment, early stopping).
   - Output artifact: `checkpoints/<exp>/finetune_best.pt` (encoder +
     classifier).
   - Reports clean test accuracy as the *main downstream result*.

4. **Stage C — Robustness + TTT evaluation**
   - Inputs: `finetune_best.pt`, all 14 corruptions × 5 severities of
     CIFAR-10-C, plus the clean test set.
   - For each (corruption, severity) computes **baseline** (no
     adaptation) and, when `ttt.enabled` and the corresponding
     `eval_mode` flag is on, **per-batch TTT** on the full set and/or
     **per-image TTT** on a stratified subsample of size
     `per_image_subsample` (Sun 2020 TTT, rotation auxiliary, K=1).
   - Output artifact: `logs/<exp>/cifar10c_results.csv` (long format —
     one row per (corruption, severity, ttt_mode)).

`CheckpointManager.reset_best()` is called at the start of every training
stage so that "best metric" tracking from Stage A doesn't bleed into
Stage B.1, etc.

---

## Stage C CSV Schema

`logs/<exp>/cifar10c_results.csv` is the main reporting artifact. The
schema is **long** (one row per (corruption, severity, ttt_mode)) so
that per-batch and per-image numbers coexist in a single dataframe
without exploding the column count.

| column | meaning |
|---|---|
| `corruption` | corruption name (e.g. `gaussian_noise`, or `clean`) |
| `severity` | 1–5, or `0` for the `clean` row |
| `ttt_mode` | `baseline` (no adaptation), `per_batch` (Sun 2020 TTT, full set) or `per_image` (Sun 2020 TTT, subsample of `n_samples`) |
| `n_samples` | number of test images that contributed to the row (full 10 000 for `baseline` and `per_batch`, `per_image_subsample` for `per_image`) |
| `accuracy` | top-1 accuracy on those `n_samples` images |
| `loss` | cross-entropy on those `n_samples` images |
| `delta_accuracy` | `accuracy - baseline_accuracy` for the same (corruption, severity); `0` for the `baseline` row itself |

There are up to three rows per (corruption, severity) plus the same
three for `clean` at severity 0 — clean numbers anchor the rest. The
"TTT helps when?" question is answered by pivoting on `ttt_mode` and
inspecting `delta_accuracy` per corruption.

---

## Regularization & Stability Knobs

These are in addition to AMP / AdamW / cosine. They mostly live in the
fine-tune stage where overfitting is the real risk.

### Label smoothing

`nn.CrossEntropyLoss(label_smoothing=0.1)` in the fine-tune trainer.
Replaces the hard one-hot target with a soft distribution: ε=0.1 is
spread uniformly over all C=10 classes, so the true class gets
1 − ε + ε/C = 0.91 and each of the other 9 gets ε/C = 0.01
(Szegedy et al. 2016 convention, which is what PyTorch implements).
Calibrates confidence and reduces overfitting in the late epochs.

### Early stopping

`BaseTrainer.fit` tracks `epochs_since_improvement` based on validation
loss with patience `early_stopping_patience` (10 for fine-tune). When the
val loss has not improved for `patience` consecutive epochs, training
breaks out of the loop. The best checkpoint is the one written via
`save_best`, not the last epoch.

### Gradient clipping (under AMP)

`BaseTrainer._backward_step` does, in order:

1. `scaler.scale(loss).backward()`
2. `scaler.unscale_(optimizer)` — needed before clipping so the norm
   reflects true gradients, not scaled ones
3. `torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)` — default
   `grad_clip = 1.0`
4. `scaler.step(optimizer)` → `scaler.update()`

Skipping the unscale step before clipping silently makes the clip ineffective
when AMP is on.

### Stochastic depth (drop_path)

Wired through `ViTBackboneBuilder.drop_path_rate` (default 0.1 for the
overnight config). Inside each transformer block, residual branches are
randomly dropped with linearly increasing probability up the depth.
Cheap, strong regularizer for transformers.

---

## Training Hardware

The reported results were produced on **Google Colab** with an
**NVIDIA A100-SXM4-40GB** (SXM4, 40 GB HBM2e, CUDA 13.0, driver 580.82.07).
AMP (bfloat16/float16) was enabled for all stages. Total wall-clock time for
the full default pipeline (200 ep SimCLR + 30 ep linear probe + 30 ep
fine-tune + Stage C eval) was approximately 2–3 hours on this hardware.

---

## Config Presets & Notebook Switch

Two YAML configs live in `configs/`:

- **`configs/default.yaml`** — overnight run on A100. SimCLR 200 ep
  (batch 1024), fine-tune 30 ep, linear probe 30 ep, full Stage C eval.
  Approx. 2–3 h on an A100-SXM4-40GB.
- **`configs/smoke.yaml`** — 2 epochs per stage, batch 256/128, used for
  validating the pipeline end-to-end in roughly 5 minutes before
  committing to an overnight run.

The Colab notebook (`notebooks/colab.ipynb`) selects between them via a
`CONFIG_PRESET` variable in cell 2:

```python
CONFIG_PRESET = "smoke"     # or "default"
CONFIG_PATHS = {
    "default": "configs/default.yaml",
    "smoke":   "configs/smoke.yaml",
}
```

`CONFIG_OVERRIDES` in the same cell can patch any individual field
without editing YAML — useful for quick LR / batch-size sweeps from the
notebook.

**Recommended flow before any overnight launch:** run `smoke` first, see
that all four stages complete and the CSV is written, then flip back to
`default` and launch.

---

## References

Papers and codebases the project builds on.

### Architecture

- **ViT** — Dosovitskiy, A. *et al.* "An Image is Worth 16x16 Words:
  Transformers for Image Recognition at Scale." *ICLR 2021.*
  [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- **timm library** — Wightman, R. "PyTorch Image Models." 2019.
  [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
  (provides the `VisionTransformer` implementation used in
  `src/models/backbone.py`).

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
  Reference implementation: `github.com/yueatsprograms/ttt_cifar_release`.

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

- **Label smoothing** — Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens,
  J., Wojna, Z. "Rethinking the Inception Architecture for Computer
  Vision." *CVPR 2016.*
  [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
- **Stochastic depth (drop_path)** — Huang, G., Sun, Y., Liu, Z.,
  Sedra, D., Weinberger, K. "Deep Networks with Stochastic Depth."
  *ECCV 2016.* [arXiv:1603.09382](https://arxiv.org/abs/1603.09382)
- **RandAugment** — Cubuk, E. D., Zoph, B., Shlens, J., Le, Q. V.
  "RandAugment: Practical Automated Data Augmentation with a Reduced
  Search Space." *NeurIPS 2020.*
  [arXiv:1909.13719](https://arxiv.org/abs/1909.13719)
- **Random Erasing** — Zhong, Z., Zheng, L., Kang, G., Li, S., Yang, Y.
  "Random Erasing Data Augmentation." *AAAI 2020.*
  [arXiv:1708.04896](https://arxiv.org/abs/1708.04896)

### Normalization layers (background)

- **LayerNorm** — Ba, J. L., Kiros, J. R., Hinton, G. E. "Layer
  Normalization." 2016.
  [arXiv:1607.06450](https://arxiv.org/abs/1607.06450).
  ViT uses LayerNorm; the encoder being adapted at test time inherits
  these layers.
- **BatchNorm** — Ioffe, S., Szegedy, C. "Batch Normalization:
  Accelerating Deep Network Training by Reducing Internal Covariate
  Shift." *ICML 2015.*
  [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- **GroupNorm** — Wu, Y., He, K. "Group Normalization." *ECCV 2018.*
  [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)

### Frameworks

- **PyTorch** — Paszke, A. *et al.* "PyTorch: An Imperative Style,
  High-Performance Deep Learning Library." *NeurIPS 2019.*
  [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)
