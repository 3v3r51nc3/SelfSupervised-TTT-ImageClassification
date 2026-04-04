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
- temperature τ = 0.5 (from config) — lower = sharper distribution,
  model is forced to be more precise about which pairs are similar
- batch size = 128 → 254 negatives per pair (more negatives = better signal)
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

TTT fixes this by doing a few gradient steps on each test image before predicting:
1. Take a test image (no label available)
2. Apply a self-supervised objective (e.g. SimCLR loss on two augmented views)
3. Update a small part of the model (only normalization layers = norm_only)
4. Predict the class with the updated model
5. Reset weights for the next image

This way the model adapts locally to each test sample without needing labels.

**Why norm_only?**
LayerNorm layers (γ, β parameters) control the scale and shift of activations.
They are sensitive to distribution shift and quick to adapt.
Updating only them is fast and avoids destroying the learned representations.

For the TER objective, TTT must be evaluated **without access to test labels**
and compared against the same model **without TTT**. The key question is not
just whether the model predicts well, but whether this test-time adaptation
actually helps under distribution shift.

---

## CIFAR-10 and CIFAR-10-C

60 000 images total, 10 classes (airplane, car, bird, cat, deer,
dog, frog, horse, ship, truck), 32×32 pixels, RGB.

Split used in this project:
- 45 000 train (SSL pretraining + supervised fine-tune)
- 5 000 validation (hyperparameter tuning)
- 10 000 test (final evaluation)

CIFAR-10-C is the same test set but with 19 types of corruption
(noise, blur, weather effects) at 5 severity levels. In this project it is not
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
