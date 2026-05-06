# Han et al. 2025 — When TTA Meets Self-Supervised Models

**Reference**
Han, J., Park, J., Han, D., Hwang, W.
*When Test-Time Adaptation Meets Self-Supervised Models.*
[arXiv:2506.23529](https://arxiv.org/abs/2506.23529), 2025.

Local PDF: [`When_Test_Time_Adaptation_Meets_Self_Supervised_Models_arXiv_2506_23529_2025_compressed.pdf`](../pdfs/When_Test_Time_Adaptation_Meets_Self_Supervised_Models_arXiv_2506_23529_2025_compressed.pdf)

---

## Core idea

Systematic study of the **interaction between modern TTA methods** (TENT, EATA, SAR, etc.) **and self-supervised pretrained models** (DINO, MAE, MoCo, SimCLR). Asks the question: are SSL pretrained models intrinsically more *adaptable* than their supervised counterparts?

## Key findings

- SSL models produce **more zero-shot robust representations** than supervised models on moderate shifts.
- Conversely, TTA methods designed for supervised models (e.g. TENT) **bring less marginal improvement** on SSL backbones — there is less headroom.
- Architecture choice (ViT vs CNN) and normalization (BN vs LN) are decisive factors.

## Why this reference for us

- **Third reference in the TER subject** ([3]) — recent (2025), useful for positioning the project.
- Empirically confirms our observation that the combination **ViT + LN + classical TTT (Sun 2020)** is not optimal — Han 2025 documents exactly this friction.

## Discussion angle

Useful when explaining why per-batch gains saturate around +1.5 pp at steps=5: classical TTA on SSL models often has less headroom than on supervised models. This is consistent with our setup.
