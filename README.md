# Dual-Axis Out-of-Distribution (OOD) Detection on Molecular Graphs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/RDKit-2024.03+-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A benchmark and data-centric framework for evaluating **Graph Neural Networks (GNNs)** under **Dual-Axis Out-of-Distribution (OOD)** shifts. This repository formalizes OOD detection into two orthogonal chemical axes: **Structural / Syntactic Validity** (physical valency violations) and **Semantic / Scaffold Novelty** (domain shift into unseen chemical chemotypes).

---

## 📌 The Dual-Axis Formulation

Traditional OOD detection in machine learning collapses all distributional anomalies into a single scalar score. In molecular chemistry, however, failure modes are multifaceted:

```
                          AXIS 1: Structural Validity
                          (Syntactic / Chemical Rules)
                                   ▲
                                   │
              Chemically Invalid   │   Invalid Novel Graphs
              (Corrupted Graphs)   │   (e.g., Texas Carbons)
                                   │
      ─────────────────────────────┼─────────────────────────────►
          In-Distribution (ID)     │   AXIS 2: Semantic Novelty
          (Known Scaffolds)        │   (New Scaffolds / Chemotypes)
                                   │
                                   ▼
```

- **Axis 1 — Structural Validity (Syntactic OOD):** Evaluates whether a molecular graph conforms to physical chemistry laws. Negative samples are chemically corrupted graphs containing **pentavalent "Texas" carbons** that strictly fail RDKit sanitization.
- **Axis 2 — Semantic Novelty (Semantic OOD):** Evaluates model reliability when encountering valid chemical structures with entirely unseen core scaffolds (partitioned via **Bemis-Murcko scaffold splitting**).

---

## 🔬 Dataset Overview (Tox21)

We build upon the **Tox21** (Toxicology in the 21st Century) bioactivity benchmark:
- **Raw Count:** 7,831 compounds across 12 toxicological bioassay targets.
- **Preprocessing:** Validated and sanitized using RDKit; 8 unparseable inorganic compounds (e.g., aluminum hydrides) were removed.
- **Curated Dataset:** **7,823 clean molecules**.

### Scaffold-Based Splitting (70 / 10 / 10 / 10)
To prevent **analog bias** (where identical ring systems leak between train and test sets), molecules were clustered by their **Bemis-Murcko Scaffold**:

| Split | Molecules | Role | Description |
|---|---|---|---|
| `id_train` | 5,476 (~70%) | In-Distribution Train | Primary training partition for GNN representations |
| `id_val` | 782 (~10%) | In-Distribution Validation | Hyperparameter optimization & threshold tuning |
| `id_test` | 783 (~10%) | In-Distribution Test | Generalization baseline on known scaffolds |
| `ood_valid` | 782 (~10%) | **Axis 2: Semantic OOD** | 100% disjoint scaffold cores never seen during training |

---

## ⚡ Controlled Graph Corruption (Axis 1)

To create negative samples for structural validity, we developed `add_bond_overvalent_carbon_v3`:
1. Identifies candidate pairs of neutral, non-aromatic aliphatic carbon atoms ($sp^3, sp^2$) that do not share a bond.
2. Inserts an explicit single bond between them, driving at least one carbon's total valence to **5** (exceeding carbon's physical limit of 4).
3. Explicit hydrogens are frozen (`explicit_hs`, `no_implicit = True`) to prevent automatic RDKit hydrogen adjustment.
4. **Validation:** Reconstructed from JSON and validated via `Chem.SanitizeMol()`. **100% of generated invalid graphs fail sanitization.**

Because standard SMILES cannot represent invalid molecules without throwing errors, corrupted graphs are serialized as raw node/edge JSON topologies ready for PyTorch Geometric / DGL ingestion.

---

## 📊 Exploratory Data Analysis (EDA) & Validation

All exploratory data analysis and validation routines are reproducible via `eda/eda_analysis.py` and `eda/EDA_Dual_Axis_OOD.ipynb`.

### Key Validation Findings

1. **Zero Data Leakage (11/11 Checks Passed):**
   - 0 SMILES overlap between all split pairs.
   - 0 scaffold hash overlap between ID sets and `ood_valid`.
   - Strict 1-to-1 provenance mapping for corrupted graphs.

2. **No Shortcut Exploitation (Hypothesis Testing):**
   - We evaluated whether OOD detection could be solved by trivial size baselines (e.g., counting atoms or molecular weight).
   - **Mann-Whitney U tests** and **Cohen's $d$ effect sizes** across 6 physicochemical descriptors confirm that all differences are negligible (**Cohen's $d < 0.11$**). TPSA shows no significant difference ($p = 0.793$).

3. **Chemical Balance:**
   - Elemental frequencies (C: ~74%, O: ~14%, N: ~8%) and bond types remain consistent across ID and OOD splits.

---

## 📁 Repository Structure

```
Dual-Axis OOD Detection/
├── Dataset/                               # Cleaned splits and corrupted graph JSONs
│   ├── manifest.json                      # Checksums, record counts, and metadata
│   ├── id_train_clean.csv                 # 5,476 ID training molecules
│   ├── id_val_clean.csv                   # 782 ID validation molecules
│   ├── id_test_clean.csv                  # 783 ID test molecules
│   ├── ood_valid_clean.csv                # 782 Axis 2 Semantic OOD molecules
│   ├── id_train_corrupted_v3_graphs.json  # Axis 1 invalid graphs (Train pool)
│   ├── id_val_corrupted_v3_graphs.json    # Axis 1 invalid graphs (Val pool)
│   ├── id_test_corrupted_v3_graphs.json   # Axis 1 invalid graphs (Test pool)
│   └── ...                                # Pilot audits and validation records
├── eda/                                   # Data quality & validation suite
│   ├── plots/                             # 10 publication-ready figures
│   │   ├── 01_molecule_sizes.png
│   │   ├── 02_atom_type_distribution.png
│   │   ├── 03_bond_type_distribution.png
│   │   ├── 04_scaffold_distribution.png
│   │   ├── 05_corruption_rates.png
│   │   ├── 06_shortcut_analysis.png
│   │   ├── 07_leakage_audit.png
│   │   ├── 08_label_distribution.png
│   │   ├── 09_valid_vs_invalid_size.png
│   │   └── 10_overvalent_distribution.png
│   ├── EDA_Dual_Axis_OOD.ipynb            # Interactive 12-section EDA notebook
│   ├── eda_analysis.py                    # Standalone reproducible CLI script
│   ├── leakage_audit.csv                  # 11-point leakage audit table
│   ├── shortcut_analysis.csv              # Statistical tests & Cohen's d table
│   ├── descriptive_statistics.csv         # Molecular properties summary
│   ├── evaluation_metrics.json            # Standardized evaluation protocol
│   └── data_quality_report.md             # Formal Data Quality Lead report
├── data_preparation.ipynb                 # Original dataset curation notebook
├── summary_till_eda.md                    # Complete project briefing & viva prep guide
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
└── README.md                              # Repository documentation
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/dual-axis-ood-detection.git
cd dual-axis-ood-detection
pip install -r requirements.txt
```

### 2. Run EDA & Verification Suite
To reproduce all statistical tests, audits, and regenerate all figures:
```bash
python eda/eda_analysis.py
```

### 3. Open Interactive Notebooks
```bash
jupyter lab
```
- Open `eda/EDA_Dual_Axis_OOD.ipynb` to inspect interactive visualizations.
- Open `data_preparation.ipynb` to view the original scaffold splitting and corruption logic.

---

## 📈 Standardized Evaluation Protocol

Models evaluated on this benchmark should report the following metrics across both axes:

| Metric | Full Name | Purpose |
|---|---|---|
| **AUROC** | Area Under ROC Curve | Threshold-independent discrimination capacity |
| **AUPR-In** | Area Under Precision-Recall (ID as positive) | Performance under prevalence imbalance |
| **AUPR-Out** | Area Under Precision-Recall (OOD as positive) | Anomaly detection effectiveness |
| **FPR@95%TPR** | False Positive Rate at 95% True Positive Rate | Industry safety standard: false alarms at 95% detection |

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
