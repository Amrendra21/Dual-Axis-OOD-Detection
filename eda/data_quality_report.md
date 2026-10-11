# Dual-Axis OOD Detection — EDA & Data Quality Report

**Role:** Data Quality Lead  
**Date:** 2026-10-11  
**Dataset:** Tox21 (7,823 valid molecules + 4,195 invalid graphs)

---

## 1. Dataset Overview

| Split | Type | Count | % of Valid |
|---|---|---|---|
| `id_train` | Valid (ID) | 5,476 | 70.0% |
| `id_val` | Valid (ID) | 782 | 10.0% |
| `id_test` | Valid (ID) | 783 | 10.0% |
| `ood_valid` | Valid (OOD, scaffold-held-out) | 782 | 10.0% |
| `invalid_train` | Corrupted graphs | 3,237 | — |
| `invalid_val` | Corrupted graphs | 481 | — |
| `invalid_test` | Corrupted graphs | 477 | — |
| **Total valid** | | **7,823** | |
| **Total invalid** | | **4,195** | |
| **Source removed** | AlH3 molecules | 8 | |

---

## 2. Molecular Size & Physicochemical Distributions

![Molecule Sizes](file:///C:/Users/Amrendra%20Singh/.gemini/antigravity-ide/brain/30c257e6-69b8-456d-9a4b-d38b3d5543b9/plots/01_molecule_sizes.png)

### Descriptive Statistics

| Split | N | Atom Mean | Atom Median | Atom Std | Bond Mean | MW Mean | Rings Mean | % Cyclic | LogP Mean | TPSA Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| ID Train | 5,476 | 18.44 | 16.0 | 11.33 | 19.13 | 275.31 | 1.81 | 77.6% | 2.37 | 59.20 |
| ID Val | 782 | 18.55 | 16.0 | 11.14 | 19.24 | 272.60 | 1.81 | 76.7% | 2.27 | 62.16 |
| ID Test | 783 | 18.38 | 16.0 | 10.66 | 19.12 | 270.94 | 1.79 | 78.4% | 2.38 | 58.32 |
| OOD Valid | 782 | 19.69 | 18.0 | 11.92 | 20.61 | 290.76 | 1.96 | 74.7% | 2.53 | 59.87 |
| Invalid Train | 3,237 | 21.92 | 20.0 | 12.52 | 23.91 | — | — | — | — | — |
| Invalid Val | 481 | 21.53 | 20.0 | 12.11 | 23.47 | — | — | — | — | — |
| Invalid Test | 477 | 21.55 | 20.0 | 11.31 | 23.56 | — | — | — | — | — |

> [!NOTE]
> Invalid graphs are larger on average than their valid counterparts because they only come from molecules that **could** be corrupted (requires at least one suitable non-aromatic carbon) — small/acyclic/fully-aromatic molecules are excluded.

---

## 3. Atom & Bond Type Distributions

![Atom Types](file:///C:/Users/Amrendra%20Singh/.gemini/antigravity-ide/brain/30c257e6-69b8-456d-9a4b-d38b3d5543b9/plots/02_atom_type_distribution.png)

![Bond Types](file:///C:/Users/Amrendra%20Singh/.gemini/antigravity-ide/brain/30c257e6-69b8-456d-9a4b-d38b3d5543b9/plots/03_bond_type_distribution.png)

**Findings:**
- Carbon dominates all splits (expected for drug-like organic chemistry)
- Atom type profiles are **highly consistent** across ID Train, Val, Test, and OOD Valid
- Nitrogen and Oxygen are the next most common; halogen distribution (F, Cl, Br) is consistent
- Bond type ratios (aromatic/single/double) are nearly identical across all splits
- **No systematic atom/bond-type bias between ID and OOD** — good for fair evaluation

---

## 4. Scaffold Distribution

![Scaffold Distribution](file:///C:/Users/Amrendra%20Singh/.gemini/antigravity-ide/brain/30c257e6-69b8-456d-9a4b-d38b3d5543b9/plots/04_scaffold_distribution.png)

**Scaffold Diversity:**

| Split | Molecules | Unique Scaffolds | Diversity (scaffolds/mol) |
|---|---|---|---|
| ID Train | 5,476 | ~2,100+ | High |
| ID Val | 782 | High | High |
| ID Test | 783 | High | High |
| OOD Valid | 782 | All unseen | **Zero overlap with ID** |

- The most common scaffold in ID Train is benzene (`c1ccccc1`), as expected for Tox21
- OOD scaffolds are completely disjoint from all ID scaffolds (verified in leakage audit)
- Acyclic molecules use canonical SMILES as scaffold key — no degenerate "no-scaffold" bucket

---

## 5. Shortcut Analysis — Can a Naive Model Cheat?

> **Question:** Do OOD molecules differ from ID molecules on simple, learnable statistics (atom count, MW, LogP, etc.)? If yes, a naive baseline could achieve high OOD detection AUROC without learning true distribution shift.

![Shortcut Analysis](file:///C:/Users/Amrendra%20Singh/.gemini/antigravity-ide/brain/30c257e6-69b8-456d-9a4b-d38b3d5543b9/plots/06_shortcut_analysis.png)

### Statistical Tests (Mann-Whitney U, two-sided)

| Metric | ID Mean | OOD Mean | ID Median | OOD Median | p-value | Cohen's d | Effect Size | Shortcut Risk |
|---|---|---|---|---|---|---|---|---|
| Atom Count | 18.44 | 19.69 | 16.0 | 18.0 | 0.00099 | -0.107 | **Negligible** | Low |
| Molecular Weight | 274.52 | 290.76 | 238.3 | 266.3 | 0.00145 | -0.097 | **Negligible** | Low |
| Ring Count | 1.81 | 1.96 | 1.0 | 2.0 | 0.00042 | -0.049 | **Negligible** | Low |
| Aromatic Atom Count | 6.27 | 5.87 | 6.0 | 5.0 | 0.00546 | +0.063 | **Negligible** | Low |
| LogP | 2.36 | 2.53 | 2.35 | 2.62 | 0.02394 | -0.075 | **Negligible** | Low |
| TPSA | 59.43 | 59.88 | 46.5 | 47.9 | 0.791 | -0.008 | **Negligible** | **None** |

### Conclusion

> [!IMPORTANT]
> While some metrics are statistically significant (p < 0.05), **all effect sizes are negligible** (|Cohen's d| < 0.2). The OOD molecules are only slightly larger/heavier on average, but the overlap in distributions is almost complete. A naive size-threshold baseline **cannot reliably separate** ID from OOD. The scaffold-based split correctly isolates structural novelty without introducing gross physicochemical bias.

**Verdict: No meaningful shortcut — the benchmark is fair.**

---

## 6. Corruption Pipeline Audit

![Corruption Rates](file:///C:/Users/Amrendra%20Singh/.gemini/antigravity-ide/brain/30c257e6-69b8-456d-9a4b-d38b3d5543b9/plots/05_corruption_rates.png)

### Corruption Success Rates

| Split | Attempted | Successful | Failed | Success Rate |
|---|---|---|---|---|
| Train | 5,476 | 3,237 | 2,239 | **59.11%** |
| Val | 782 | 481 | 301 | **61.51%** |
| Test | 783 | 477 | 306 | **60.92%** |
| **Total** | **7,041** | **4,195** | **2,846** | **59.58%** |

**Why ~40% fail?**
Corruption requires a molecule to have at least one non-aromatic, neutral carbon atom that becomes over-valent upon bond addition. Molecules that are fully aromatic, have only sp2 carbons, or lack eligible carbon atoms cannot be corrupted by this method. Failure rates are **consistent across splits** (~59–62%), indicating no systematic bias.

### Corrupted Atom Valence

![Valid vs Invalid](file:///C:/Users/Amrendra%20Singh/.gemini/antigravity-ide/brain/30c257e6-69b8-456d-9a4b-d38b3d5543b9/plots/10_overvalent_distribution.png)

- 100% of `explicit_valence_i` values are **≥ 5** (over-valent carbon)
- Confirms the corruption is always chemically invalid

### Reconstruction Validation

| Split | Validated | Sanitization Failures | Errors | Failure Rate |
|---|---|---|---|---|
| Train | 3,237 | 3,237 | 0 | **100%** |
| Val | 481 | 481 | 0 | **100%** |
| Test | 477 | 477 | 0 | **100%** |

✅ **All 4,195 invalid graphs confirmed chemically invalid via RDKit round-trip reconstruction.**

---

## 7. Leakage Audit

![Leakage Audit](file:///C:/Users/Amrendra%20Singh/.gemini/antigravity-ide/brain/30c257e6-69b8-456d-9a4b-d38b3d5543b9/plots/07_leakage_audit.png)

| Check | Result | Status |
|---|---|---|
| Mol-ID cross-split leakage (valid mols) | 0 | ✅ PASS |
| Scaffold overlap: OOD vs ID | **0** | ✅ PASS |
| Canonical SMILES overlap: OOD vs ID | **0** | ✅ PASS |
| Within-split duplicates: id_train | 0 | ✅ PASS |
| Within-split duplicates: id_val | 0 | ✅ PASS |
| Within-split duplicates: id_test | 0 | ✅ PASS |
| Within-split duplicates: ood_valid | 0 | ✅ PASS |
| Invalid graph duplicates: inv_train | 0 | ✅ PASS |
| Invalid graph duplicates: inv_val | 0 | ✅ PASS |
| Invalid graph duplicates: inv_test | 0 | ✅ PASS |
| Invalid graph cross-split leakage | 0 | ✅ PASS |

> [!IMPORTANT]
> **All 11 leakage checks pass.** Zero scaffold overlap between OOD and ID splits confirms that the novelty detection task cannot be solved by memorization. Zero SMILES overlap rules out near-duplicate entries.

---

## 8. Valid vs Corrupted Graph Size Comparison

![Valid vs Invalid Size](file:///C:/Users/Amrendra%20Singh/.gemini/antigravity-ide/brain/30c257e6-69b8-456d-9a4b-d38b3d5543b9/plots/09_valid_vs_invalid_size.png)

> [!NOTE]
> Invalid graphs are slightly larger than their source valid molecules because they are drawn only from the subset of molecules that have corruptible carbons (larger molecules have more candidates). This is a **known selection bias** but does not undermine the structural invalidity task — a model cannot exploit size alone because the difference is small and overlaps heavily.

---

## 9. Bioactivity Label Distribution

![Label Distribution](file:///C:/Users/Amrendra%20Singh/.gemini/antigravity-ide/brain/30c257e6-69b8-456d-9a4b-d38b3d5543b9/plots/08_label_distribution.png)

**Key observations:**
- Positive rates are generally low (0–30%), consistent with Tox21's class imbalance
- `SR-MMP` has the highest positive rate across all splits
- `NR-AR-LBD`, `NR-PPAR-gamma` have the lowest positive rates
- Label distributions are **consistent across ID splits** — scaffold splitting preserved label balance
- OOD Valid has **similar label rates** to ID splits — no systematic label-based shortcut

---

## 10. Standardized Evaluation Metrics

### Axis 1 — Structural Validity Detection
*Task: Classify valid graphs (negative) vs corrupted graphs (positive)*

| Metric | Description | Direction |
|---|---|---|
| **AUROC** | Area under ROC curve | Higher is better |
| **AUPR** | Area under PR curve (invalid = positive) | Higher is better |
| **FPR@95TPR** | False positive rate at 95% true positive rate | Lower is better |
| **Detection Accuracy** | Balanced accuracy at optimal threshold | Higher is better |

### Axis 2 — Semantic Novelty Detection
*Task: Classify ID molecules (negative) vs OOD molecules (positive)*

Same four metrics; positive class = `ood_valid`.

### Combined Dual-Axis Score
```
Dual-Axis AUROC = (AUROC_axis1 + AUROC_axis2) / 2
```

### Result File Standard
```csv
model_name, split, axis, metric, value, seed, timestamp
GIN_baseline, id_test, axis1_validity, AUROC, 0.923, 42, 2026-10-11T12:00:00
GIN_baseline, ood_valid, axis2_novelty, AUROC, 0.781, 42, 2026-10-11T12:00:00
```

---

## 11. Data Quality Summary

```
=== MOLECULE COUNTS ===
Total valid molecules            : 7,823
  ID splits (train/val/test)     : 7,041
  OOD Valid                      : 782
Invalid graphs generated         : 4,195 (59.58% success rate)
Source molecules removed         : 8 (aluminium-containing SMILES)

=== LEAKAGE AUDIT ===
All 11 leakage checks            : PASS (zero violations)
  - Scaffold overlap OOD/ID      : 0
  - SMILES overlap OOD/ID        : 0
  - Cross-split mol-ID leakage   : 0
  - Within-split duplicates      : 0 (all 4 splits)

=== CORRUPTION PIPELINE ===
100% of invalid graphs confirmed : RDKit sanitization FAIL
Reconstruction errors            : 0
Valence of corrupted atom        : 100% >= 5 (over-valent)
Consistent failure rate          : ~59-62% across all splits

=== SHORTCUT ANALYSIS ===
All physicochemical metrics      : Negligible effect size (|d| < 0.2)
Shortcut risk                    : LOW — no exploitable simple baseline

=== STATUS ===
DATA QUALITY                     : PASS
READY FOR MODEL TRAINING         : YES
```

---

## 12. Deliverables Checklist

| Deliverable | Status | Location |
|---|---|---|
| EDA notebook | ✅ Done | `eda/EDA_Dual_Axis_OOD.ipynb` |
| Molecule size plots | ✅ Done | `eda/plots/01_molecule_sizes.png` |
| Atom type plots | ✅ Done | `eda/plots/02_atom_type_distribution.png` |
| Bond type plots | ✅ Done | `eda/plots/03_bond_type_distribution.png` |
| Scaffold distribution plots | ✅ Done | `eda/plots/04_scaffold_distribution.png` |
| Corruption audit plot | ✅ Done | `eda/plots/05_corruption_rates.png` |
| Shortcut analysis plot | ✅ Done | `eda/plots/06_shortcut_analysis.png` |
| Leakage audit plot | ✅ Done | `eda/plots/07_leakage_audit.png` |
| Label distribution plot | ✅ Done | `eda/plots/08_label_distribution.png` |
| Valid vs invalid size plot | ✅ Done | `eda/plots/09_valid_vs_invalid_size.png` |
| Over-valent distribution plot | ✅ Done | `eda/plots/10_overvalent_distribution.png` |
| Leakage audit CSV | ✅ Done | `eda/leakage_audit.csv` |
| Descriptive statistics CSV | ✅ Done | `eda/descriptive_statistics.csv` |
| Shortcut analysis CSV | ✅ Done | `eda/shortcut_analysis.csv` |
| Evaluation metrics JSON | ✅ Done | `eda/evaluation_metrics.json` |
| This data quality report | ✅ Done | (this document) |
