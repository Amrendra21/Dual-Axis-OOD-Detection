# Dual-Axis Molecule Dataset (Version 3)
### Disentangling Structural Validity from Semantic Novelty in Graph OOD Detection

[![Dataset Status](https://img.shields.io/badge/Dataset%20Build-PASSED-brightgreen.svg)](processed_v3/BUILD_REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-ee4c2c.svg)](https://pytorch.org/)

---

## 🗂️ Repository Structure

```
Dual-Axis OOD Detection/
│
├── README.md                                 ← This file
├── LICENSE                                   ← MIT License
├── .gitignore
│
├── docs/                                     ← Project documentation
│   ├── DATA_CARD.md                          ← Comprehensive dataset documentation & evaluation rules
│   ├── Final Report The Dual-Axis Molecule Dataset (Version 3).pdf
│   └── The Whole Project.docx
│
├── audits/                                   ← Independent audit deliverables
│   ├── dataset_integrity_report.md           ← Integrity: counts, shapes, leakage, twin mapping
│   ├── corruption_validation_report.md       ← Corruption families A/B/C validation
│   ├── novelty_analysis.md                   ← Scaffold vs. dataset novelty analysis
│   └── notebooks/
│       └── shortcut_audit.ipynb              ← Executable shortcut AUROC reproduction notebook
│
└── processed_v3/                             ← Dataset package (pre-built PyTorch tensors)
    ├── BUILD_REPORT.md                       ← Build statistics & shortcut audit warnings
    ├── config.json                           ← Build config, seeds, SHA-256 checksums
    ├── feature_schema.json                   ← Feature definitions (30 atom, 6 bond features)
    ├── graphs/                               ← Pre-computed graph tensor files (.pt)
    │   ├── id_train.pt                       ← 5,432 in-distribution training graphs
    │   ├── id_val.pt                         ← 679 in-distribution validation graphs
    │   ├── id_test.pt                        ← 679 in-distribution test graphs
    │   ├── novel_scaffold_val.pt             ← 408 held-out scaffold validation graphs
    │   ├── novel_scaffold_test.pt            ← 347 held-out scaffold test graphs
    │   ├── novel_dataset_val.pt              ← 225 SIDER dataset-shift validation graphs
    │   ├── novel_dataset_test.pt             ← 198 SIDER dataset-shift test graphs
    │   ├── corrupt_train.pt                  ← 1,086 corrupted training graphs (Families A, B)
    │   ├── corrupt_val.pt                    ← 543 corrupted validation graphs (Families A, B)
    │   └── corrupt_test.pt                   ← 543 corrupted test graphs (Families A, B, C)
    ├── tables/                               ← Metadata & mapping tables (CSV)
    │   ├── master_table.csv                  ← 10,140 rows × 33 columns (SMILES, splits, labels)
    │   ├── eligible_pairs.csv                ← Corrupted graph → valid twin mapping (2,172 pairs)
    │   ├── matched_controls.csv              ← Novelty target → size-matched ID control (1,178 pairs)
    │   └── dropped_rows.csv                  ← Curation log of filtered molecules
    ├── splits/
    │   └── split_ids.json                    ← JSON mapping of split names to graph UIDs
    ├── audits/
    │   └── audit_results.json                ← Quantitative build audit metrics & shortcut baselines
    └── logs/                                 ← Build pipeline logs
```

---

## 🎯 Conceptual Framework

Traditional Out-of-Distribution (OOD) detection in graph learning compresses all distributional anomalies into a single scalar uncertainty score. However, molecular graphs exhibit fundamentally decoupled failure modes:

```
                          AXIS 1: Structural Validity
                          (Syntactic / Chemical Valence Rules)
                                   ▲
                                   │
              Chemically Invalid   │   Invalid Novel Graphs
              (Corrupted Graphs)   │   (e.g., Hypervalent Edits)
                                   │
      ─────────────────────────────┼─────────────────────────────►
          In-Distribution (ID)     │   AXIS 2: Semantic Novelty
          (Known Scaffolds)        │   (Held-Out Scaffolds & Datasets)
                                   │
                                   ▼
```

- **Axis 1 — Structural Validity (Syntactic OOD):** Evaluates whether a molecular graph conforms to physical valency rules. Tested using targeted synthetic valence corruptions:
  - **Family A (Bond Inflation):** Inflates non-aromatic single/double bond orders. Perfectly count-invariant (clean validity test).
  - **Family B (Bond Relocation):** Moves one end of a bond to a saturated atom (degree shift).
  - **Family C (Atom Swap — Held Out):** Swaps carbon atoms for oxygen atoms (evaluated in test only).
- **Axis 2 — Semantic Novelty (Semantic OOD):** Evaluates model generalization on chemically valid molecules with unseen topological features:
  - **Scaffold Novelty (`novel_scaffold`):** Tox21 molecules with Bemis-Murcko cyclic frameworks completely held-out from training.
  - **Dataset Shift Novelty (`novel_dataset`):** Approved and clinical drug molecules from SIDER, de-overlapped against Tox21.
  - **Size-Matched Controls (`matched_controls.csv`):** Caliper-matched ID controls ($|\Delta \text{atoms}| \le 2, |\Delta \text{bonds}| \le 2$) to eliminate size confounding.

---

## 📦 Dataset Quick Stats

| Split | Category | Graphs |
|:---|:---|---:|
| `id_train` | In-Distribution (Tox21) | 5,432 |
| `id_val` | In-Distribution (Tox21) | 679 |
| `id_test` | In-Distribution (Tox21) | 679 |
| `novel_scaffold_val` | Semantic Novelty — Scaffold | 408 |
| `novel_scaffold_test` | Semantic Novelty — Scaffold | 347 |
| `novel_dataset_val` | Semantic Novelty — Dataset Shift (SIDER) | 225 |
| `novel_dataset_test` | Semantic Novelty — Dataset Shift (SIDER) | 198 |
| `corrupt_train` | Syntactic Corruption (Fam. A & B) | 1,086 |
| `corrupt_val` | Syntactic Corruption (Fam. A & B) | 543 |
| `corrupt_test` | Syntactic Corruption (Fam. A, B & C) | 543 |
| **Total** | | **10,140** |

---

## ⚡ Loading the Dataset in Python

Graphs are stored as lists of dictionaries, readable with `torch.load`:

```python
import torch

# Load in-distribution training set
train_graphs = torch.load("processed_v3/graphs/id_train.pt", weights_only=True)
sample = train_graphs[0]

print("UID:",           sample["uid"])            # e.g., 'T21-00000'
print("Node features:", sample["x"].shape)        # [N, 30]
print("Edge index:",    sample["edge_index"].shape)# [2, E] (bidirectional)
print("Edge features:", sample["edge_attr"].shape) # [E, 6]
print("Labels (y):",    sample["y"].shape)         # [1, 12] (NaN for unlabeled tasks)
```

---

## 📑 Audit Deliverables

All audit artifacts are located in [`audits/`](audits/), independently generated directly from the saved disk tensors:

| File | Description |
|:---|:---|
| [`audits/dataset_integrity_report.md`](audits/dataset_integrity_report.md) | Graph counts, tensor shapes, non-finite values, twin mapping, SMILES/scaffold leakage |
| [`audits/corruption_validation_report.md`](audits/corruption_validation_report.md) | Family A/B/C corruption verification: edit counts, valence violations, pair deltas |
| [`audits/novelty_analysis.md`](audits/novelty_analysis.md) | Scaffold vs. dataset novelty analysis; caliper matching protocol validation |
| [`audits/notebooks/shortcut_audit.ipynb`](audits/notebooks/shortcut_audit.ipynb) | Executable shortcut AUROC reproduction (RF + LR count-only baselines) |

---

## 🔬 Benchmark Recommendations for Evaluators

1. **Validity Axis:** Always report AUROC per corruption family (**Family A**, **Family B**, **Family C**) rather than aggregating them into a single score. Family A is the gold standard for valence reasoning because summary count models achieve chance performance (~0.50).
2. **Novelty Axis:** Always evaluate novelty AUROCs against **`processed_v3/tables/matched_controls.csv`**. Size matching eliminates gross size artifacts (dropping AUROC by 7–10 points) so that model scores reflect representation learning of novel chemotypes rather than simple atom counting.
3. **Count-Only Baseline:** Publications should always report a count-only tabular baseline (e.g., Random Forest on vertex/element histograms) alongside GNN architectures.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
