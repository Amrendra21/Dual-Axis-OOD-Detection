# Dual-Axis Molecular OOD Detection

> **Disentangling Structural Validity from Semantic Novelty in Graph Neural Networks**  
> *A rigorously audited benchmark, multi-axis detection pipeline, and empirical investigation of molecular out-of-distribution detection.*

[![Build Status](https://img.shields.io/badge/Dataset%20Build-PASSED-brightgreen?style=flat-square)](processed_v3/BUILD_REPORT.md)
[![Step 9 Audit](https://img.shields.io/badge/Step%209%20Audit-VERIFIED%20(15%2F15%20Exact)-success?style=flat-square)](step9_audit/STEP9_FINAL_AUDIT_REPORT.md)
[![Regression Tests](https://img.shields.io/badge/Regression%20Tests-6%2F6%20SUITES%20PASS-brightgreen?style=flat-square)](step8_validation/regression_test_results.txt)
[![Checkpoints](https://img.shields.io/badge/Frozen%20Checkpoints-VERIFIED%20(embed__dim=256)-blue?style=flat-square)](checkpoints/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2022.09%2B-blue?style=flat-square)](https://www.rdkit.org/)

---

## Table of Contents

1. [Executive Overview](#executive-overview)
2. [The Two Evaluation Axes](#the-two-evaluation-axes)
3. [Dataset Architecture (processed_v3)](#dataset-architecture-processed_v3)
4. [Step-by-Step Research & Development Journey (Steps 1–9)](#step-by-step-research--development-journey-steps-19)
5. [Key Empirical Results & Benchmarks](#key-empirical-results--benchmarks)
6. [Core Scientific Insights](#core-scientific-insights)
7. [Model Checkpoints & Artifact Verification](#model-checkpoints--artifact-verification)
8. [Repository Structure](#repository-structure)
9. [Quick Start & Reproducibility](#quick-start--reproducibility)
10. [Project Team & Responsibilities](#project-team--responsibilities)

---

## Executive Overview

Standard graph-level out-of-distribution (OOD) detection benchmarks collapse all distributional shifts into a single scalar score. In molecular chemistry, however, distribution shifts occur along two **orthogonal failure axes**:

| Failure Axis | Real-World Meaning | Chemical Example | Failure Mode if Conflated |
|:---|:---|:---|:---|
| **Axis 1: Structural Validity (Syntactic OOD)** | Graph violates fundamental chemical physics and valence rules. | A carbon atom with 5 covalent bonds ($CH_5^+$ octet violation). | Models exploit atom/degree counts rather than chemical grammar. |
| **Axis 2: Semantic Novelty (Semantic OOD)** | Graph is physically valid but lies outside the chemical training manifold. | A legitimate bioactive compound possessing an unseen ring scaffold or foreign disease domain. | Distance-to-train estimators conflate molecule size with genuine scaffold novelty. |

Treating these axes as one leads models to exploit superficial shortcuts (such as molecule size or degree distributions). This repository provides:
- A curated, leak-free benchmark (`processed_v3`, 10,140 graphs) with caliper-matched controls.
- A 4-layer GINE toxicity backbone encoder producing 256-dimensional embeddings.
- A GINE-based structural validity detector achieving $>0.995$ AUROC across both seen and held-out corruption families.
- A regularized Mahalanobis novelty estimator and an ECFP4 fingerprint Tanimoto baseline.
- An extensive 9-step progression documenting model fitting, independent auditing, statistical validation, and final paper-readiness verification.

---

## The Two Evaluation Axes

```
                                  MOLECULAR GRAPH INPUT
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
         [AXIS 1: STRUCTURAL VALIDITY]                 [AXIS 2: SEMANTIC NOVELTY]
              Is it physically valid?                       Is it chemically novel?
                     │                                             │
      ┌──────────────┴──────────────┐               ┌──────────────┴──────────────┐
      ▼                             ▼               ▼                             ▼
Family A: Bond Inflation     Family C: Atom Swap  Novel Scaffold:            Novel Dataset:
(Count-invariant gold test)   (Held-out test fam)  Bemis-Murcko ring shifts  SIDER domain shift
      │                             │               │                             │
      ▼                             ▼               └──────────────┬──────────────┘
Family B: Bond Relocation    GINE Detector                         ▼
(Degree histogram shift)     (AUROC > 0.995)            Caliper-Matched Controls
                                                        (|Δatoms| ≤ 2, |Δbonds| ≤ 2)
```

### Axis 1 — Structural Validity (Syntactic OOD)
Tests whether models can detect physically impossible valence violations. Synthetic, targeted edits are paired with valid twins:
- **Family A — Bond Order Inflation:** Non-aromatic single/double bond inflated. **Count-invariant:** Atom counts, bond counts, and element compositions are identical to the valid twin. A count-only Random Forest scores $\text{AUROC} \approx 0.50$ (chance).
- **Family B — Bond Relocation:** Bond disconnected and re-attached to a saturated atom, inducing a degree histogram shift in ~77–81% of pairs.
- **Family C — Atom Swap (Held-Out Test Set Only):** Carbon replaced with oxygen, shifting element compositions in 100% of pairs. Completely unseen during training and validation to evaluate zero-shot corruption generalization.

### Axis 2 — Semantic Novelty (Semantic OOD)
Tests whether models can detect chemically valid yet unfamiliar molecular manifolds:
- **Novel Scaffold:** Molecules from Tox21 whose cyclic ring topology (Bemis-Murcko scaffold) has zero overlap with `id_train`.
- **Novel Dataset:** Approved drug molecules from SIDER, testing covariate and functional group domain shifts.
- **Caliper-Matched Controls (`matched_controls.csv`):** In-distribution molecules matched on size ($|\Delta\text{atoms}| \le 2, |\Delta\text{bonds}| \le 2$) to isolate topological/chemical shift from trivial atom-count correlations.

---

## Dataset Architecture (processed_v3)

**Total:** 10,140 molecular graphs across 10 deterministic splits:

| Split Identifier | Category | Count | Ground-Truth Labels |
|:---|:---|---:|:---|
| `id_train` | In-distribution — Training | 5,432 | 12 multi-task Tox21 endpoints (~16% missing) |
| `id_val` | In-distribution — Validation | 679 | 12 multi-task Tox21 endpoints (~20% missing) |
| `id_test` | In-distribution — Final Test | 679 | 12 multi-task Tox21 endpoints (~20% missing) |
| `novel_scaffold_val` | Semantic Novelty — Held-out scaffold (Val) | 408 | 12 Tox21 endpoints (zero scaffold overlap with train) |
| `novel_scaffold_test` | Semantic Novelty — Held-out scaffold (Test) | 347 | 12 Tox21 endpoints (zero scaffold overlap with train) |
| `novel_dataset_val` | Semantic Novelty — SIDER domain shift (Val) | 225 | Non-Tox21 source (`NaN` toxicity labels) |
| `novel_dataset_test` | Semantic Novelty — SIDER domain shift (Test) | 198 | Non-Tox21 source (`NaN` toxicity labels) |
| `corrupt_train` | Structural Corruption — Fam. A & B (Train) | 1,086 | Synthetically corrupted (twin mapped via `eligible_pairs.csv`) |
| `corrupt_val` | Structural Corruption — Fam. A & B (Val) | 543 | Synthetically corrupted (272 Fam A, 271 Fam B) |
| `corrupt_test` | Structural Corruption — Fam. A, B & C (Test) | 543 | 181 Fam A, 181 Fam B, 181 held-out Fam C |
| **Total** | | **10,140** | |

**Feature Dimensions:**
- Node features $x \in \mathbb{R}^{N \times 30}$: 13 element one-hot, 6 degree indicators, 4 formal charge states, 5 hydrogen counts, aromaticity, ring membership.
- Edge index $\in \mathbb{R}^{2 \times E}$: Bidirectional molecular graph topology.
- Edge attributes $e \in \mathbb{R}^{E \times 6}$: Single, double, triple, aromatic, other bond types, in-ring indicator.
- Multi-task targets $y \in \mathbb{R}^{1 \times 12}$: 12 Tox21 toxicity endpoints (`NaN` denotes unassayed targets).

---

## Step-by-Step Research & Development Journey (Steps 1–9)

The project progressed through nine rigorous phases:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             PROJECT ROADMAP (STEPS 1–9)                          │
├─────────────────┬─────────────────┬─────────────────┬────────────────────────────┤
│ Step 1          │ Step 2          │ Step 3          │ Step 4                     │
│ Data Loader &   │ Toxicity GINE   │ Validity GINE   │ Mahalanobis Novelty        │
│ Schema Check    │ Backbone (256d) │ Detector        │ Estimator (id_train)       │
├─────────────────┼─────────────────┼─────────────────┼────────────────────────────┤
│ Step 5          │ Step 6          │ Step 7          │ Step 8                     │
│ Score Pipeline  │ Independent     │ ECFP4 Finger-   │ Reproducibility &          │
│ & Evaluation    │ Audit & Scaff.  │ print Baselines │ Statistical Tests          │
├─────────────────┴─────────────────┴─────────────────┴────────────────────────────┤
│ Step 9: Final Scientific Audit, Calibration, & Robustness Gate (Ready for Paper) │
│ Handoff: Decoupled Benchmark Package to Member 5 for External Submission         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Step 1 — Graph Dataset Loading & Verification
- **Code:** [`src/data.py`](src/data.py), [`tests/test_loader.py`](tests/test_loader.py).
- Implemented `MolecularDataset` and sequential batch collator preserving 1-to-1 graph UID order (`T21-xxxxx`, `COR-xxxxx`, `SID-xxxxx`).
- Guaranteed strict mask-handling for missing multi-task Tox21 labels (`NaN` preservation in loss computations).
- Verified tensor shapes ($[N, 30]$ node features, $[E, 6]$ edge features, $[1, 12]$ label vector).

### Step 2 — Toxicity GNN Backbone Training
- **Code:** [`src/toxicity_model.py`](src/toxicity_model.py), [`src/train_toxicity.py`](src/train_toxicity.py), [`docs/GNN_TRAINING_REPORT.md`](docs/GNN_TRAINING_REPORT.md).
- **Architecture:** 4-layer Graph Isomorphism Network with Edge features (GINE), `embed_dim = 256`, combining mean + sum graph pooling.
- **Training:** Trained on `id_train` (5,432 graphs) with masked multi-task BCE loss for 100 epochs with AdamW and Cosine Annealing.
- **Outcome:** Best checkpoint at Epoch 19 achieves **Val Macro ROC-AUC = 0.8291** across the 12 endpoints. Saved at [`checkpoints/toxicity_gine/best_model.pt`](checkpoints/toxicity_gine/best_model.pt).

### Step 3 — Structural Validity Detector
- **Code:** [`src/validity_model.py`](src/validity_model.py), [`src/train_validity.py`](src/train_validity.py), [`docs/VALIDITY_DETECTION_REPORT.md`](docs/VALIDITY_DETECTION_REPORT.md).
- **Architecture:** 4-layer GINE backbone (`embed_dim = 256`) coupled to a 2-layer MLP classification head outputting a scalar validity score $\in [0, 1]$.
- **Training:** Trained on `id_train` (negative class = 0) vs. `corrupt_train` (positive class = 1, Families A & B) using binary cross-entropy. Family C remained strictly unseen.
- **Outcome:** Best checkpoint at Epoch 6 achieves **Val Family A AUROC = 0.9966** and **Val Family B AUROC = 0.9952**. Saved at [`checkpoints/validity_gine/best_validity_model.pt`](checkpoints/validity_gine/best_validity_model.pt).

### Step 4 — Regularized Mahalanobis Novelty Estimator
- **Code:** [`src/novelty_model.py`](src/novelty_model.py), [`src/fit_novelty.py`](src/fit_novelty.py), [`docs/NOVELTY_ESTIMATOR_REPORT.md`](docs/NOVELTY_ESTIMATOR_REPORT.md).
- Extracted 256-dimensional pooled graph embeddings from the frozen toxicity backbone on `id_train` ($N = 5,432$).
- Fitted empirical mean vector $\mu \in \mathbb{R}^{256}$ and covariance matrix $\Sigma \in \mathbb{R}^{256 \times 256}$ using **Ledoit-Wolf shrinkage** ($\alpha = 0.003779$).
- Regularized distance metric: $d_M(z) = \sqrt{(z - \mu)^T \Sigma_{\text{reg}}^{-1} (z - \mu)}$.
- Exported frozen estimator parameters to [`checkpoints/novelty_mahal/estimator.npz`](checkpoints/novelty_mahal/estimator.npz).

### Step 5 — Score Integration & Baseline Pipeline Evaluation
- **Code:** [`src/evaluate.py`](src/evaluate.py), [`src/run_eval.py`](src/run_eval.py), [`src/run_final_eval.py`](src/run_final_eval.py), [`docs/STEP5_EVALUATION_REPORT.md`](docs/STEP5_EVALUATION_REPORT.md).
- Integrated all three frozen components into an end-to-end evaluation pipeline with deterministic UID ordering.
- **Crucial Diagnostic Finding:**
  - Validity detection performed near-perfectly (>0.995 AUROC).
  - Novelty detection on held-out scaffolds showed an apparent failure: raw AUROC = 0.4566, matched AUROC = 0.4783 (worse than random guessing).
  - Investigation revealed that `mean_sum` pooling coupled with distance-to-train estimators induced a strong size bias: `id_val` molecules averaged $22.5$ atoms while `id_train` averaged $16.8$ atoms ($r = 0.694$ between atom count and Mahalanobis score).
- Test set evaluation was executed once under strict freezing authorization, producing [`predictions/val_predictions.csv`](predictions/val_predictions.csv) and [`predictions/test_predictions.csv`](predictions/test_predictions.csv).

### Step 6 — Independent Audit & Novelty Improvement Experiments
- **Code:** [`step6_audit/`](step6_audit/), [`step6_audit/STEP6_AUDIT_REPORT.md`](step6_audit/STEP6_AUDIT_REPORT.md).
- Investigated the root causes of scaffold novelty performance and benchmarked 8 alternative estimators on validation splits (zero test tuning):
  1. *Scaffold-Disjointness Phenomenon:* Tox21 was split by Murcko scaffold, meaning `id_val` is scaffold-disjoint from `id_train`. Thus, validation in-distribution molecules also feature unseen scaffolds relative to training.
  2. *Alternative Novelty Methods Benchmarked:* Layer-wise Mahalanobis, Mean-only pooling, Multi-task energy score, and ECFP4 distance.
  3. Proved that under size-matched controls, all distance-to-train representations hover between 0.46–0.51 AUROC.

### Step 7 — Molecular Novelty Detection via ECFP4 Fingerprints
- **Code:** [`step7_novelty/`](step7_novelty/), [`step7_novelty/STEP7_NOVELTY_REPORT.md`](step7_novelty/STEP7_NOVELTY_REPORT.md), [`tests/test_step7_novelty.py`](tests/test_step7_novelty.py).
- Evaluated whether traditional chemoinformatic representations outperform GNN latent spaces for novelty detection.
- Computed 2048-bit Morgan fingerprints (radius = 2) and defined novelty as minimum Tanimoto distance to `id_train`: $d(m) = 1 - \max_{t \in \text{train}} T(m, t)$.
- **Empirical Result:** On scaffold novelty with matched controls, ECFP4 achieves **0.4950 AUROC**, which is statistically indistinguishable from GNN Mahalanobis (0.4783 AUROC, $p = 0.559$). However, on true domain shift (SIDER), GNN Mahalanobis superiorly detects covariate shift (raw 0.6074 vs. ECFP4 0.4943).

### Step 8 — Reproducibility, Statistical Validation, & Paper Readiness
- **Code:** [`step8_validation/`](step8_validation/), [`step8_validation/STEP8_REPRODUCIBILITY_REPORT.md`](step8_validation/STEP8_REPRODUCIBILITY_REPORT.md).
- Complete reproduction of the 6-suite regression tests.
- Re-computed all metrics from scratch, confirming identical numerical outputs.
- Performed rigorous statistical hypothesis tests:
  - DeLong tests for AUROC differences.
  - 2,000-fold bootstrap 95% confidence intervals.
  - Permutation tests confirming that scaffold novelty differences are statistically insignificant ($p > 0.40$).
  - Verified validity detector invariance to acyclic shortcuts ($r = -0.0162, p = 0.571$).

### Step 9 — Final Scientific Audit & Paper-Readiness Gate
- **Code:** [`step9_audit/`](step9_audit/), [`step9_audit/STEP9_FINAL_AUDIT_REPORT.md`](step9_audit/STEP9_FINAL_AUDIT_REPORT.md).
- Executed an independent verification of all frozen checkpoints, embeddings, and metrics.
- Re-evaluated all 15 validation and test metrics directly from raw `.pt` graphs and frozen weights. **Zero discrepancies observed ($\Delta = 0.000000$).**
- Evaluated probability calibration: Expected Calibration Error ($\text{ECE} = 0.0215$) and confidence separation ($0.897$).
- Confirmed zero-shot generalization to held-out **Family C corruption: 0.9968 AUROC** ($\text{FPR95} = 0.000$).

---

## Key Empirical Results & Benchmarks

### 1. Structural Validity Detection (Axis 1)

Evaluated using the frozen GINE Validity Detector ([`checkpoints/validity_gine/best_validity_model.pt`](checkpoints/validity_gine/best_validity_model.pt)). Higher score = higher corruption probability.

| Split | Benchmark Setting | Negative Split ($N$) | Positive Split ($N$) | AUROC | AUPR | FPR95 | Status |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|
| **Val** | All Corruptions (A + B) | `id_val` (679) | `corrupt_val` (543) | **0.9959** | **0.9961** | **0.0044** | ✅ Verified |
| **Val** | **Family A (Count-Invariant)** | `id_val` (679) | `corrupt_val_fam_a` (272) | **0.9966** | **0.9933** | **0.0103** | ✅ Gold Standard |
| **Val** | Family B (Degree Shift) | `id_val` (679) | `corrupt_val_fam_b` (271) | **0.9952** | **0.9934** | **0.0044** | ✅ Verified |
| **Test** | All Corruptions (A + B + C)| `id_test` (679) | `corrupt_test` (543) | **0.9958** | **0.9965** | **0.0000** | ✅ Verified |
| **Test** | **Family A (Count-Invariant)** | `id_test` (679) | `corrupt_test_fam_a` (181) | **0.9986** | **0.9961** | **0.0000** | ✅ Gold Standard |
| **Test** | Family B (Degree Shift) | `id_test` (679) | `corrupt_test_fam_b` (181) | **0.9921** | **0.9885** | **0.0000** | ✅ Verified |
| **Test** | **Family C (Held-out Unseen)**| `id_test` (679) | `corrupt_test_fam_c` (181) | **0.9968** | **0.9942** | **0.0000** | ✅ Zero-Shot OOD |

### 2. Semantic Novelty Detection (Axis 2)

Comparison between GNN Mahalanobis distance, ECFP4 Tanimoto minimum distance, and Multi-Task Energy score.

| Benchmark | Setting | Estimator Method | AUROC | 95% Conf. Interval | AUPR | FPR95 |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **Novel Scaffold (Val)** | Raw (vs. `id_val`) | GNN Mahalanobis (Frozen) | 0.4566 | [0.421, 0.492] | 0.3440 | 0.9632 |
| **Novel Scaffold (Val)** | **Matched Controls** | **GNN Mahalanobis (Frozen)** | **0.4783** | [0.436, 0.520] | **0.4956** | **0.9866** |
| **Novel Scaffold (Val)** | Matched Controls | ECFP4 Tanimoto ($r=2$) | 0.4950 | [0.453, 0.537] | 0.5119 | 0.9543 |
| **Novel Scaffold (Val)** | Matched Controls | Multi-Task Energy Score | 0.5044 | [0.463, 0.546] | 0.5165 | 0.9677 |
| **Novel Dataset (Val)** | Raw (vs. `id_val`) | GNN Mahalanobis (Frozen) | 0.6074 | [0.565, 0.649] | 0.3347 | 0.9116 |
| **Novel Dataset (Val)** | **Matched Controls** | **GNN Mahalanobis (Frozen)** | **0.5362** | [0.480, 0.592] | **0.5291** | **0.9526** |
| **Novel Dataset (Val)** | Matched Controls | ECFP4 Tanimoto ($r=2$) | 0.4652 | [0.409, 0.521] | 0.4480 | 0.9573 |
| **Novel Dataset (Val)** | Matched Controls | Multi-Task Energy Score | 0.5208 | [0.465, 0.577] | 0.5186 | 0.9431 |
| **Novel Scaffold (Test)** | Raw (vs. `id_test`)| GNN Mahalanobis (Frozen) | 0.5013 | [0.464, 0.539] | 0.3467 | 0.9278 |
| **Novel Scaffold (Test)** | **Matched Controls** | **GNN Mahalanobis (Frozen)** | **0.4536** | [0.408, 0.499] | **0.4874** | **0.9876** |
| **Novel Dataset (Test)** | Raw (vs. `id_test`) | GNN Mahalanobis (Frozen) | 0.6660 | [0.624, 0.708] | 0.3590 | 0.8071 |
| **Novel Dataset (Test)** | **Matched Controls** | **GNN Mahalanobis (Frozen)** | **0.5730** | [0.514, 0.632] | **0.5726** | **0.9144** |

### 3. Shortcut Benchmark Comparison (GNN vs. Count Baseline)

A Random Forest trained purely on 21 tabular graph summary statistics (atom counts, bond counts, element compositions, degree bins) demonstrates that Family A requires true structural graph reasoning:

| Evaluation Split / Family | Tabular Count-Only AUROC | GINE Validity Detector AUROC | Margin Above Shortcut |
|:---|:---:|:---:|:---:|
| Val / **Family A (Count-Invariant)** | 0.493 (Chance) | **0.9966** | **+0.504** (Pure Graph Learning) |
| Val / Family B (Degree Shift) | 0.652 | **0.9952** | **+0.343** |
| Test / **Family A (Count-Invariant)**| 0.489 (Chance) | **0.9986** | **+0.510** (Pure Graph Learning) |
| Test / Family B (Degree Shift) | 0.652 | **0.9921** | **+0.340** |
| Test / **Family C (Held-out Unseen)**| 0.798 | **0.9968** | **+0.199** |

---

## Core Scientific Insights

1. **Orthogonal Axes Require Separate Detection Mechanisms:**
   Structural validity is a local chemical-physics constraint (bond valences, octet rules), which message-passing GNNs learn almost flawlessly (>0.995 AUROC). Semantic novelty is a global manifold density problem, which standard distance-to-train estimators struggle to capture under scaffold splits. Conflating both into one score conceals this fundamental dichotomy.

2. **The Scaffold-Disjointness Phenomenon:**
   When datasets are partitioned using Bemis-Murcko scaffolds to create a realistic OOD benchmark, `id_val` molecules also possess unseen scaffolds relative to `id_train`. Therefore, in-distribution validation molecules are *also* novel scaffolds. A pure distance-to-train estimator cannot distinguish in-distribution scaffold novelty from target scaffold novelty under size-matched controls ($\text{AUROC} \approx 0.48–0.50, p > 0.40$).

3. **Size Confounding in Graph Embeddings:**
   Standard global pooling operations (particularly `sum` and `mean_sum` pooling) scale monotonically with molecule size. In raw splits, larger validation molecules receive higher novelty scores simply because of atom count ($r = 0.694$). Using caliper-matched controls is essential to prevent false claims of semantic novelty detection.

4. **Chemoinformatic vs. Learned Representations:**
   ECFP4 Morgan fingerprints and GNN latent embeddings face the identical performance ceiling on scaffold novelty under matched controls ($0.4950$ vs. $0.4783, p = 0.559$). However, on domain shift (SIDER), GNN representations significantly outperform fingerprints ($0.5362$ vs. $0.4652, p = 0.056$).

---

## Model Checkpoints & Artifact Verification

All frozen models operate in $\mathbb{R}^{256}$ and have been independently audited:

| Component | Architecture / Method | File Path | File Size | SHA-256 Checksum |
|:---|:---|:---|:---:|:---|
| **Toxicity Backbone** | 4-layer GINE, mean+sum pool | [`checkpoints/toxicity_gine/best_model.pt`](checkpoints/toxicity_gine/best_model.pt) | 2,195,004 B | `4de2f66f8932ee51d0919cd17db839208cf2b090288bddf56066f278491c4416` |
| **Validity Detector** | GINE encoder + 2-layer MLP head | [`checkpoints/validity_gine/best_validity_model.pt`](checkpoints/validity_gine/best_validity_model.pt) | 2,122,525 B | `87bc0a8edd13b54c99e3126be8a46c1c83ca990ce212ad53c04bab1deeb05420` |
| **Novelty Estimator** | Ledoit-Wolf Mahalanobis ($\alpha=0.0038$)| [`checkpoints/novelty_mahal/estimator.npz`](checkpoints/novelty_mahal/estimator.npz) | 491,285 B | `eca1da4fdfc9797bd4097e7fc5c65937a7817e89f97c78fd4624920fc2687d27` |

---

## Repository Structure

```
Dual-Axis-OOD-Detection/
│
├── README.md                          ← You are here (consolidated master overview)
├── LICENSE                            ← MIT License
├── requirements.txt                   ← Dependencies
│
├── src/                               ← Core library code
│   ├── data.py                        ← Graph dataset & batch collator
│   ├── toxicity_model.py              ← 4-layer GINE toxicity backbone architecture
│   ├── validity_model.py              ← GINE + MLP validity detector architecture
│   ├── novelty_model.py               ← MahalanobisNoveltyEstimator (Ledoit-Wolf)
│   ├── loss.py                        ← Masked BCE loss for multi-task Tox21
│   ├── train_toxicity.py              ← Training script for toxicity backbone
│   ├── train_validity.py              ← Training script for validity detector
│   ├── fit_novelty.py                 ← Fitting script for Mahalanobis estimator
│   ├── evaluate.py                    ← Metrics, scoring, and calibration functions
│   ├── run_eval.py                    ← Validation evaluation pipeline runner
│   └── run_final_eval.py              ← Protected final test evaluation runner
│
├── checkpoints/                       ← Frozen model checkpoints (audited & verified)
│   ├── toxicity_gine/                 ← best_model.pt (Epoch 19, Val AUC=0.8291)
│   ├── validity_gine/                 ← best_validity_model.pt (Epoch 6, Fam A=0.9966)
│   └── novelty_mahal/                 ← estimator.npz (Ledoit-Wolf, dim=256)
│
├── predictions/                       ← Exported model score tables & metrics
│   ├── val_predictions.csv            ← 3,213 benchmark comparison rows
│   ├── test_predictions.csv           ← 3,125 benchmark comparison rows
│   ├── val_scores_by_split.csv        ← Unique validation graph scores (1,855 rows)
│   ├── test_scores_by_split.csv       ← Unique test graph scores (1,767 rows)
│   ├── val_metrics.json               ← 7 validation AUROC / AUPR / FPR95 records
│   └── test_metrics.json              ← 8 test AUROC / AUPR / FPR95 records
│
├── processed_v3/                      ← Molecular graph dataset package (10,140 graphs)
│   ├── graphs/                        ← 10 PyTorch tensor splits (.pt)
│   ├── tables/                        ← master_table.csv, matched_controls.csv, etc.
│   └── feature_schema.json            ← Node and edge attribute definitions
│
├── step6_audit/                       ← Step 6: Root cause audit & novelty explorations
│   ├── STEP6_AUDIT_REPORT.md          ← Scaffold disjointness analysis & 8 baselines
│   └── novelty_experiment_results.json
│
├── step7_novelty/                     ← Step 7: ECFP4 fingerprint Tanimoto benchmark
│   ├── STEP7_NOVELTY_REPORT.md        ← ECFP4 vs. GNN Mahalanobis comparison
│   └── val_ecfp4_predictions.csv
│
├── step8_validation/                  ← Step 8: Reproducibility & statistical tests
│   ├── STEP8_REPRODUCIBILITY_REPORT.md← DeLong tests, bootstrap CIs, permutation tests
│   └── regression_test_results.txt    ← Full test suite execution logs
│
├── step9_audit/                       ← Step 9: Final scientific audit & paper gate
│   ├── STEP9_FINAL_AUDIT_REPORT.md    ← Comprehensive audit & calibration analysis
│   ├── step9_audit_summary.json       ← Exact 15-metric recomputation records
│   └── run_step9_audit.py             ← End-to-end audit verification script
│
├── tests/                             ← Complete regression test suite
│   ├── test_loader.py                 ← Dataset loading, shapes, and UID integrity
│   ├── test_model.py                  ← Toxicity model forward pass and loss masking
│   ├── test_validity.py               ← Validity detector loading and scoring
│   ├── test_novelty.py                ← Mahalanobis estimator math and shrinkage
│   ├── test_integration.py            ← Pipeline integration and authorization gate
│   └── test_step7_novelty.py          ← ECFP4 fingerprint pipeline tests
│
├── audits/                            ← Independent dataset audits (Member 3)
│   ├── dataset_integrity_report.md
│   ├── corruption_validation_report.md
│   └── novelty_analysis.md
│
└── docs/                              ← Comprehensive documentation reports
    ├── DATA_CARD.md                   ← Dataset specification & evaluation rules
    ├── GNN_TRAINING_REPORT.md         ← Step 2 training log and hyperparameters
    ├── VALIDITY_DETECTION_REPORT.md   ← Step 3 training log and metrics
    ├── NOVELTY_ESTIMATOR_REPORT.md    ← Step 4 fitting log and covariance checks
    └── STEP5_EVALUATION_REPORT.md     ← Step 5 initial baseline evaluation report
```

---

## Quick Start & Reproducibility

### 1. Environment Setup

```bash
git clone https://github.com/Amrendra21/Dual-Axis-OOD-Detection.git
cd Dual-Axis-OOD-Detection
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Full Test Suite

Execute the regression test suite across all 6 modules:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3. Load Frozen Checkpoints & Run Inference

```python
import torch
from src.evaluate import load_frozen_components, score_split

# Load toxicity backbone, validity detector, and Mahalanobis novelty estimator
components = load_frozen_components(
    data_dir="processed_v3",
    toxicity_ckpt_path="checkpoints/toxicity_gine/best_model.pt",
    validity_ckpt_path="checkpoints/validity_gine/best_validity_model.pt",
    novelty_ckpt_path="checkpoints/novelty_mahal/estimator.npz",
    device="cpu"
)

# Score in-distribution validation split
df_val_scores = score_split(
    split_name="id_val",
    components=components,
    data_dir="processed_v3",
    device="cpu"
)

print(df_val_scores[["uid", "validity_score", "novelty_score"]].head())
```

### 4. Reproduce Step 9 Scientific Audit

Run the automated scientific audit script to verify all 15 metrics, check calibration, and test Family C zero-shot generalization:

```bash
python step9_audit/run_step9_audit.py
```

---

## Project Team & Responsibilities

| Role | Member | Primary Responsibilities & Key Deliverables |
|:---|:---|:---|
| **Literature Review** | Anish Chaudhary | Research background, conceptual framework, related work synthesis. |
| **Dataset Construction** | Aanjneya Nayak | Graph building, synthetic corruption engine, Bemis-Murcko splitting (`processed_v3`). |
| **Dataset Audit** | Amrendra Singh | Independent dataset verification, shortcut analysis, caliper matching audit (`audits/`). |
| **Model Development & Audits** | Aryan Singh | GNN training, validity detector, novelty estimators, Steps 1–9 execution, paper-readiness gates. |
| **Evaluation & Paper Integration** | Aditya Kanaujiya | Per-molecule prediction handoff, external validation, paper drafting and integration. |

---

## License

This project is open-source and licensed under the [MIT License](LICENSE).
