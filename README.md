# Dual-Axis Molecule Dataset (Version 3)
### Disentangling Structural Validity from Semantic Novelty in Graph OOD Detection

[![Dataset Status](https://img.shields.io/badge/Dataset%20Build-PASSED-brightgreen.svg)](processed_v3/BUILD_REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-ee4c2c.svg)](https://pytorch.org/)

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

## 📦 Dataset Package (`processed_v3/`)

The primary dataset is pre-computed as PyTorch tensor graphs and standardized metadata tables:

```
processed_v3/
├── graphs/                                # Pre-computed PyTorch tensor graph dictionaries
│   ├── id_train.pt                        # 5,432 in-distribution training graphs
│   ├── id_val.pt                          # 679 in-distribution validation graphs
│   ├── id_test.pt                         # 679 in-distribution test graphs
│   ├── novel_scaffold_val.pt              # 408 held-out scaffold validation graphs
│   ├── novel_scaffold_test.pt             # 347 held-out scaffold test graphs
│   ├── novel_dataset_val.pt               # 225 SIDER dataset shift validation graphs
│   ├── novel_dataset_test.pt              # 198 SIDER dataset shift test graphs
│   ├── corrupt_train.pt                   # 1,086 corrupted training graphs (Families A, B)
│   ├── corrupt_val.pt                     # 543 corrupted validation graphs (Families A, B)
│   └── corrupt_test.pt                    # 543 corrupted test graphs (Families A, B, and C)
├── tables/
│   ├── master_table.csv                   # Master table (10,140 rows) with SMILES, splits, and 12 Tox21 labels
│   ├── eligible_pairs.csv                 # Corrupted graph -> valid twin graph mapping (same split)
│   ├── matched_controls.csv               # Novelty target -> size-matched ID control mapping
│   └── dropped_rows.csv                   # Curation log of filtered molecules
├── splits/
│   └── split_ids.json                     # JSON mapping of split names to graph UIDs
├── audits/
│   └── audit_results.json                 # Quantitative build audit metrics and shortcut baselines
├── feature_schema.json                    # Feature definitions (30 atom features, 6 bond features)
├── DATA_CARD.md                           # Comprehensive dataset documentation & evaluation rules
├── BUILD_REPORT.md                        # Verification and build statistics
└── config.json                            # Build configuration, seeds, and SHA-256 hashes
```

---

## 📑 Independent Audit Reports & Notebooks

All audit artifacts have been independently generated, reproduced directly from the saved disk tensors, and validated:

1. **[`dataset_integrity_report.md`](dataset_integrity_report.md):**  
   Independent verification of all 10,140 graphs, tensor dimensions ($[N, 30]$ nodes, $[2, E]$ edges, $[E, 6]$ attributes, $[1, 12]$ labels), zero non-finite values, label missingness distributions, twin relationship consistency, and zero SMILES/scaffold leakage.

2. **[`shortcut_audit.ipynb`](shortcut_audit.ipynb):**  
   Fully executable, pre-rendered Jupyter Notebook independently reproducing count-only AUROC baselines (Random Forest & Logistic Regression) across validity and novelty axes, comparing against Member 2's reported results, and illustrating size-matching effects.

3. **[`corruption_validation_report.md`](corruption_validation_report.md):**  
   Detailed evaluation of synthetic valence corruptions: Family A (bond inflation), Family B (bond relocation), and Family C (held-out atom swap). Quantifies edit distributions ($1-3$ edits), valence violation counts, and pair deltas.

4. **[`novelty_analysis.md`](novelty_analysis.md):**  
   Detailed analysis of scaffold novelty vs. dataset shift. Validates the caliper matching protocol in `matched_controls.csv` (zero violations across 1,178 pairs), characterizes unmatched graphs, and separates pure size confounding from true chemical heteroatom differences.

---

## ⚡ Loading the Dataset in Python

Graphs are stored as list of dictionaries readable with `torch.load`:

```python
import torch

# Load in-distribution training set
train_graphs = torch.load("processed_v3/graphs/id_train.pt", weights_only=True)
sample = train_graphs[0]

print("UID:", sample["uid"])                    # e.g., 'T21-00000'
print("Node features (x):", sample["x"].shape)  # [N, 30]
print("Edge index:", sample["edge_index"].shape)# [2, E] (bidirectional)
print("Edge features:", sample["edge_attr"].shape) # [E, 6]
print("Labels (y):", sample["y"].shape)          # [1, 12] (NaN for unlabeled tasks)
```

---

## 🔬 Benchmark Recommendations for Evaluators

1. **Validity Axis:** Always report AUROC per corruption family (**Family A**, **Family B**, **Family C**) rather than aggregating them into a single score. Family A is the true gold standard for valence reasoning because summary count models achieve chance performance ($\approx 0.50$).
2. **Novelty Axis:** Always evaluate novelty AUROCs against **`tables/matched_controls.csv`**. Size matching eliminates gross size artifacts (dropping AUROC by $7-10$ points) so that model scores reflect representation learning of novel chemotypes rather than simple atom counting.
3. **Count-Only Baseline:** Publications should always report a count-only tabular baseline (e.g., Random Forest on vertex/element histograms) alongside GNN architectures.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
