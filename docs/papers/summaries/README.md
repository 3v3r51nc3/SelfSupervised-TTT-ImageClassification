# Paper Summaries

One markdown summary per paper referenced in `docs/notes/tech_notes.md`. Each summary covers: full citation, core idea, key results, and how the paper is used in this project.

PDFs live in `../pdfs/`. Each summary links directly to its PDF.

---

## TTT methods (TER subject references)

| # | Paper | PDF |
|---|---|---|
| [01](01_Sun2020_TTT.md) | Sun et al. 2020 — Test-Time Training | [pdf](../pdfs/Test_time_training_with_self_supervision_for_generalization_under_distribution_shifts.pdf) |
| [02](02_Liu2021_TTTpp.md) | Liu et al. 2021 — TTT++ | [pdf](../pdfs/TTT_When_does_self_supervised_test_time_training_fail_or_thrive.pdf) |
| [03](03_Han2025_TTA_meets_SSL.md) | Han et al. 2025 — TTA meets SSL Models | [pdf](../pdfs/When_Test_Time_Adaptation_Meets_Self_Supervised_Models_arXiv_2506_23529_2025_compressed.pdf) |
| [04](04_Mirza2023_ActMAD.md) | Mirza et al. 2023 — ActMAD | [pdf](../pdfs/Actmad_Activation_matching_to_align_distributions_for_test_time_training_compressed.pdf) |

## Building blocks

| # | Paper | PDF |
|---|---|---|
| [05](05_Chen2020_SimCLR.md) | Chen et al. 2020 — SimCLR | [pdf](../pdfs/SimCLR_Chen2020_2002.05709.pdf) |
| [06](06_Dosovitskiy2021_ViT.md) | Dosovitskiy et al. 2021 — Vision Transformer | [pdf](../pdfs/ViT_Dosovitskiy2021_2010.11929.pdf) |
| [07](07_Hendrycks2019_CIFAR10C.md) | Hendrycks & Dietterich 2019 — CIFAR-10-C | [pdf](../pdfs/CIFAR10C_Hendrycks2019_1903.12261.pdf) |
| [08](08_Krizhevsky2009_CIFAR10.md) | Krizhevsky 2009 — CIFAR-10/100 (tech report) | [pdf](../pdfs/CIFAR10_Krizhevsky2009.pdf) |
| [09](09_Gidaris2018_RotationPretext.md) | Gidaris et al. 2018 — Rotation pretext | [pdf](../pdfs/Rotation_Gidaris2018_1803.07728.pdf) |
| [10](10_Wang2021_TENT.md) | Wang et al. 2021 — TENT | [pdf](../pdfs/TENT_Wang2021_2006.10726.pdf) |

## Optimization & scheduling

| # | Paper | PDF |
|---|---|---|
| [11](11_Loshchilov2019_AdamW.md) | Loshchilov & Hutter 2019 — AdamW | [pdf](../pdfs/AdamW_Loshchilov2019_1711.05101.pdf) |
| [12](12_Loshchilov2017_SGDR.md) | Loshchilov & Hutter 2017 — SGDR (cosine annealing) | [pdf](../pdfs/SGDR_Loshchilov2017_1608.03983.pdf) |
| [13](13_Goyal2017_LinearWarmup.md) | Goyal et al. 2017 — Linear warmup | [pdf](../pdfs/LinearWarmup_Goyal2017_1706.02677.pdf) |
| [14](14_Micikevicius2018_MixedPrecision.md) | Micikevicius et al. 2018 — Mixed Precision (AMP) | [pdf](../pdfs/MixedPrecision_Micikevicius2018_1710.03740.pdf) |

## Regularization

| # | Paper | PDF |
|---|---|---|
| [15](15_Szegedy2016_LabelSmoothing.md) | Szegedy et al. 2016 — Label Smoothing (Inception v3) | [pdf](../pdfs/LabelSmoothing_Szegedy2016_InceptionV3_1512.00567.pdf) |
| [16](16_Huang2016_StochasticDepth.md) | Huang et al. 2016 — Stochastic Depth (DropPath) | [pdf](../pdfs/StochasticDepth_Huang2016_1603.09382.pdf) |
| [17](17_Cubuk2020_RandAugment.md) | Cubuk et al. 2020 — RandAugment | [pdf](../pdfs/RandAugment_Cubuk2020_1909.13719.pdf) |
| [18](18_Zhong2020_RandomErasing.md) | Zhong et al. 2020 — Random Erasing | [pdf](../pdfs/RandomErasing_Zhong2020_1708.04896.pdf) |

## Normalization layers (background)

| # | Paper | PDF |
|---|---|---|
| [19](19_Ba2016_LayerNorm.md) | Ba, Kiros & Hinton 2016 — Layer Normalization | [pdf](../pdfs/LayerNorm_Ba2016_1607.06450.pdf) |
| [20](20_Ioffe2015_BatchNorm.md) | Ioffe & Szegedy 2015 — Batch Normalization | [pdf](../pdfs/BatchNorm_Ioffe2015_1502.03167.pdf) |
| [21](21_Wu2018_GroupNorm.md) | Wu & He 2018 — Group Normalization | [pdf](../pdfs/GroupNorm_Wu2018_1803.08494.pdf) |

## Frameworks

| # | Paper | PDF |
|---|---|---|
| [22](22_Paszke2019_PyTorch.md) | Paszke et al. 2019 — PyTorch | [pdf](../pdfs/PyTorch_Paszke2019_1912.01703.pdf) |

---

## Coverage check

All papers listed in `docs/notes/tech_notes.md::References` have a summary here. Every summary corresponds to a PDF in `../pdfs/`. The `tech_notes.md::References` section is the authoritative source — when adding a new paper to this folder, update both the index above and `tech_notes.md`.
