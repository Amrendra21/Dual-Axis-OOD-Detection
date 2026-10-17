# Dual-Axis Molecule Dataset (Version 3)

> **Disentangling Structural Validity from Semantic Novelty in Graph-Level Out-of-Distribution Detection**

[![Build Status](https://img.shields.io/badge/Dataset%20Build-PASSED-brightgreen?style=flat-square)](processed_v3/BUILD_REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.11+](https://img.shields.io/badge/PyTorch-2.11%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2026.09-blue?style=flat-square)](https://www.rdkit.org/)

---

## Overview

Standard graph-level OOD detection collapses all types of distributional shift into a single score. But molecular graphs fail in two **fundamentally different** ways:

| Failure Mode | What it means | Example |
|:---|:---|:---|
| **Structural invalidity** | The graph is physically impossible | A carbon with 5 bonds (violates octet rule) |
| **Semantic novelty** | The graph is valid but chemically unfamiliar | A real drug with a ring scaffold never seen during training |

Treating these identically leads to models that appear to perform well — but are actually exploiting spurious signals (atom counts, size differences) rather than learning chemistry.

This project provides a carefully curated benchmark dataset that separates these two axes, along with full independent audit deliverables, shortcut baselines, and guidelines for fair evaluation.

---

## The Two Axes

### Axis 1 — Structural Validity (Syntactic OOD)

Tests whether a model can detect **chemically impossible** graphs. We inject synthetic, targeted valence-violating edits into valid molecules:

| Family | Corruption Mechanism | Count Changes | Benchmark Role |
|:---|:---|:---|:---|
| **A — Bond Inflation** | Inflate a non-aromatic single/double bond order | **None** — all counts identical | Gold standard: count-only AUROC ≈ 0.50 |
| **B — Bond Relocation** | Move one bond end onto a saturated atom | Degree histogram shifts (~77–81% of pairs) | Moderate shortcut exposure |
| **C — Atom Swap** *(test only)* | Replace C with O (held-out family) | Element counts shift (100% of pairs) | Tests generalization to unseen corruption |

Family A is the cleanest validity test: because atom counts, bond counts, degrees, and element compositions are **identical** to the valid twin, any model achieving above-chance AUROC must be processing edge features and reasoning about chemical valence.

### Axis 2 — Semantic Novelty (Semantic OOD)

Tests whether a model can detect **chemically valid but unfamiliar** molecules:

| Benchmark | Source | What shifts | Scaffold overlap with train |
|:---|:---|:---|:---|
| **Novel Scaffold** | Tox21 held-out | Cyclic ring topology (Bemis-Murcko frameworks) | **0** (guaranteed by scaffold split) |
| **Novel Dataset** | SIDER drug database | Chemical domain, functional groups, size | 31 scaffolds shared (expected; this is a covariate-shift test) |

Both novelty benchmarks include **size-matched controls** (`matched_controls.csv`) — caliper-matched ID molecules (|Δatoms| ≤ 2, |Δbonds| ≤ 2) — so that GNN performance cannot be inflated by simple atom counting. Size matching drops raw AUROC by **7–10 points**, revealing what the model actually learned.

---

## Dataset Statistics

**Total: 10,140 molecular graphs** from Tox21 (in-distribution + scaffold novelty + corruption) and SIDER (dataset shift novelty).

| Split | Category | Graphs | Labels |
|:---|:---|---:|:---|
| `id_train` | In-distribution — Train | 5,432 | 12 Tox21 tasks, ~16% missing |
| `id_val` | In-distribution — Val | 679 | 12 Tox21 tasks, ~20% missing |
| `id_test` | In-distribution — Test | 679 | 12 Tox21 tasks, ~20% missing |
| `novel_scaffold_val` | Semantic novelty — Held-out scaffold, Val | 408 | 12 Tox21 tasks, ~20% missing |
| `novel_scaffold_test` | Semantic novelty — Held-out scaffold, Test | 347 | 12 Tox21 tasks, ~22% missing |
| `novel_dataset_val` | Semantic novelty — SIDER domain shift, Val | 225 | All NaN (non-Tox21 source) |
| `novel_dataset_test` | Semantic novelty — SIDER domain shift, Test | 198 | All NaN (non-Tox21 source) |
| `corrupt_train` | Syntactic corruption — Fam. A & B, Train | 1,086 | All NaN (use `twin_uid` for labels) |
| `corrupt_val` | Syntactic corruption — Fam. A & B, Val | 543 | All NaN |
| `corrupt_test` | Syntactic corruption — Fam. A, B & C, Test | 543 | All NaN |
| **Total** | | **10,140** | |

**Graph feature dimensions:**
- Node feature matrix `x`: shape `[N, 30]` — 13 element indicators, 6 degree indicators, 4 formal charge indicators, 5 hydrogen count indicators, `is_aromatic`, `in_ring`
- Edge index `edge_index`: shape `[2, E]` — bidirectional (each bond stored twice)
- Edge attribute matrix `edge_attr`: shape `[E, 6]` — single, double, triple, aromatic, other, in_ring
- Label matrix `y`: shape `[1, 12]` — 12 Tox21 toxicity endpoints; `NaN` = missing

---

## Repository Structure

```
Dual-Axis OOD Detection/
│
├── README.md                          ← You are here
├── LICENSE                            ← MIT License
├── .gitignore
│
├── docs/                              ← Project documentation
│   ├── DATA_CARD.md                   ← Concise dataset specification & evaluation rules
│   ├── Final Report The Dual-Axis Molecule Dataset (Version 3).pdf
│   └── The Whole Project.docx         ← Plain-language guide for all team members
│
├── audits/                            ← Independent audit deliverables (Member 3)
│   ├── dataset_integrity_report.md    ← Graph counts, shapes, leakage, twin mapping
│   ├── corruption_validation_report.md← Family A/B/C corruption verification & shortcut analysis
│   ├── novelty_analysis.md            ← Scaffold vs. dataset novelty; caliper matching validation
│   └── notebooks/
│       └── shortcut_audit.ipynb       ← Executable AUROC reproduction notebook
│
└── processed_v3/                      ← Dataset package (pre-built PyTorch tensors)
    ├── BUILD_REPORT.md                ← Build statistics, shortcut audit warnings, AUROC baselines
    ├── config.json                    ← Build config, seeds, SHA-256 checksums of raw + output files
    ├── feature_schema.json            ← Column names for x, edge_attr, y
    ├── graphs/                        ← PyTorch tensor graph files (.pt)
    │   ├── id_train.pt                ← 5,432 graphs
    │   ├── id_val.pt                  ← 679 graphs
    │   ├── id_test.pt                 ← 679 graphs
    │   ├── novel_scaffold_val.pt      ← 408 graphs
    │   ├── novel_scaffold_test.pt     ← 347 graphs
    │   ├── novel_dataset_val.pt       ← 225 graphs
    │   ├── novel_dataset_test.pt      ← 198 graphs
    │   ├── corrupt_train.pt           ← 1,086 graphs
    │   ├── corrupt_val.pt             ← 543 graphs
    │   └── corrupt_test.pt            ← 543 graphs
    ├── tables/                        ← Metadata and mapping tables
    │   ├── master_table.csv           ← 10,140 rows × 33 cols (SMILES, splits, 12 labels)
    │   ├── eligible_pairs.csv         ← 2,172 corrupted graph → valid twin mappings
    │   ├── matched_controls.csv       ← 1,178 novelty target → size-matched ID control pairs
    │   └── dropped_rows.csv           ← Curation log (stage + reason for every filtered molecule)
    ├── splits/
    │   └── split_ids.json             ← JSON mapping of split name → list of graph UIDs
    ├── audits/
    │   └── audit_results.json         ← Quantitative build audit metrics & shortcut AUROC baselines
    └── logs/
        └── build.log                  ← Full pipeline build log
```

---

## Shortcut Audit Summary

A critical part of the benchmark is quantifying how much simple count-based models can achieve **without** learning chemistry. Below are the independently verified count-only AUROC baselines (5-fold cross-validated Random Forest on 21 summary statistics: atom count, bond count, 13 element counts, 6 degree bins):

### Validity Axis (AUROC; 0.50 = chance)

| Split / Family | Count-only AUROC | Shortcut Cue Present |
|:---|:---:|:---|
| train / **Family A** | **0.518** | None — pure graph structure test ✅ |
| val / **Family A** | **0.493** | None ✅ |
| test / **Family A** | **0.489** | None ✅ |
| train / **Family B** | 0.721 | Degree histogram shifts in 81% of pairs ⚠️ |
| val / **Family B** | 0.652 | Degree histogram shifts in 79% of pairs ⚠️ |
| test / **Family B** | 0.652 | Degree histogram shifts in 77% of pairs ⚠️ |
| test / **Family C** | 0.798 | Element counts shift in 100% of pairs ⚠️ |

> **Family A is the gold standard.** Count models achieve chance performance — any GNN performance above 0.50 represents genuine valence reasoning.

### Novelty Axis — Raw vs. Size-Matched AUROC

| Benchmark | Raw AUROC | Size-Matched AUROC | Size-Induced Drop |
|:---|:---:|:---:|:---:|
| `val / novel_scaffold` | 0.752 | 0.689 | **−0.063** |
| `test / novel_scaffold` | 0.755 | 0.646 | **−0.109** |
| `val / novel_dataset` | 0.728 | 0.664 | **−0.064** |
| `test / novel_dataset` | 0.693 | 0.629 | **−0.064** |

> **Always evaluate novelty AUROCs against `matched_controls.csv`.** The 7–10 point AUROC drop after size-matching shows how much raw performance is inflated by molecule size rather than true chemical novelty detection.

---

## Quick Start

### Loading a Graph Split

```python
import torch

# Load in-distribution training set
train_graphs = torch.load("processed_v3/graphs/id_train.pt", weights_only=True)

sample = train_graphs[0]
print("UID:          ", sample["uid"])             # e.g., 'T21-00000'
print("Node features:", sample["x"].shape)         # torch.Size([N, 30])
print("Edge index:   ", sample["edge_index"].shape)# torch.Size([2, E])
print("Edge features:", sample["edge_attr"].shape) # torch.Size([E, 6])
print("Labels (y):   ", sample["y"].shape)         # torch.Size([1, 12])
```

### Getting the Twin of a Corrupted Graph

```python
import pandas as pd, torch

pairs = pd.read_csv("processed_v3/tables/eligible_pairs.csv")
corrupt_test = torch.load("processed_v3/graphs/corrupt_test.pt", weights_only=True)

# Find the valid twin for a given corrupted graph
corrupt_uid = corrupt_test[0]["uid"]
twin_uid = pairs.set_index("corrupt_uid").loc[corrupt_uid, "twin_uid"]
print(f"Corrupt: {corrupt_uid}  →  Valid twin: {twin_uid}")
```

### Computing OOD Metrics Against Matched Controls

```python
import pandas as pd
from sklearn.metrics import roc_auc_score

controls = pd.read_csv("processed_v3/tables/matched_controls.csv")

# Filter to the benchmark you care about
scaffold_val_controls = controls[controls["novel_set"] == "novel_scaffold_val"]

# Your model scores (replace with actual predictions)
# novelty_scores: scores for novel_scaffold_val molecules
# control_scores: scores for size-matched id_val molecules

# auroc = roc_auc_score(labels, scores)
# Report this matched AUROC, not the raw ID-vs-novel AUROC
```

---

## Audit Deliverables

All audit artifacts in [`audits/`](audits/) were independently generated by loading raw `.pt` files from disk — no reliance on upstream pipeline code or documentation:

| Report | What was verified |
|:---|:---|
| [`dataset_integrity_report.md`](audits/dataset_integrity_report.md) | All 10,140 graph counts match across `.pt`, `split_ids.json`, and `master_table.csv`; zero non-finite values; zero SMILES leakage; zero scaffold leakage on `novel_scaffold` sets; 2,172/2,172 twin pairs valid |
| [`corruption_validation_report.md`](audits/corruption_validation_report.md) | Family A/B/C quota fulfillment (100%); edit count distributions (1–3 edits); pair-level invariance audit; shortcut cue quantification |
| [`novelty_analysis.md`](audits/novelty_analysis.md) | Caliper matching audit (0 violations in 1,178 pairs; 92.8% match rate); size difference post-matching (<0.11 atoms); residual heteroatom composition analysis; AUROC impact of size matching |
| [`shortcut_audit.ipynb`](audits/notebooks/shortcut_audit.ipynb) | Fully executable notebook: Random Forest + Logistic Regression count-only AUROC reproduction; comparison against Member 2's reported results; size-matching effects illustrated |

---

## Evaluation Rules for Downstream Benchmarking

These rules must be followed when evaluating GNN models on this benchmark:

1. **Report per-family validity AUROCs** — never aggregate Families A, B, and C into a single number. Family A is the clean test; B and C are confounded by structural and elemental shortcuts.

2. **Evaluate novelty against `matched_controls.csv`** — the matched AUROC is the honest metric. The raw ID-vs-novel AUROC inflates performance by 7–10 points due to size differences.

3. **Tune only on val splits** — `novel_scaffold_val`, `novel_dataset_val`, `corrupt_val`, `id_val`. The test splits are to be examined **once** for final reporting only.

4. **Always include a count-only baseline** — report a tabular count-only baseline (e.g., Random Forest on atom/bond/element/degree counts) so readers can quantify how much your GNN exceeds simple statistics.

5. **Label semantics for corrupted graphs** — corrupted molecules have all-`NaN` `y` labels. Use `eligible_pairs.csv` (`twin_uid`) to retrieve the valid twin's Tox21 labels if needed for auxiliary training.

6. **Label semantics for SIDER graphs** — `novel_dataset` graphs have all-`NaN` `y`. They carry no Tox21 labels by design; they are evaluated only on the OOD detection axis.

---

## Data Provenance & Integrity

| Raw Source | SHA-256 |
|:---|:---|
| `tox21.csv.gz` | `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360` |
| `sider.csv.gz` | `4cc7ccb20dded475853c583596e227116c222c6ca7dbb40159bdbe9ee3f099b9` |

Build configuration, random seed (`42`), and full pipeline SHA-256 hashes for all output files are recorded in [`processed_v3/config.json`](processed_v3/config.json).

**Curation summary:**

| Source | Raw rows | After cleaning | Dropped |
|:---|:---:|:---:|:---:|
| Tox21 | 7,831 | 7,545 | 286 (duplicates, parse failures, size out-of-range) |
| SIDER | 1,427 | 1,303 | 124 (duplicates, size out-of-range, Tox21 overlap: 880 removed) |

Duplicate Tox21 molecules were merged: agreeing labels were kept; conflicting labels were set to `NaN`. All filtering decisions are logged in `processed_v3/tables/dropped_rows.csv`.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Project Team

| Role | Responsibility |
|:---|:---|
| Member 1 | Literature review |
| Member 2 | Dataset construction (`processed_v3`) |
| Member 3 | Independent dataset audit (reports in `audits/`) |
| Member 4 | GNN model training & predictions |
| Member 5 | Evaluation, metric computation, paper integration |
