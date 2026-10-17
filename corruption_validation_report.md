# Dual-Axis Molecule Dataset (processed_v3): Synthetic Corruption Validation Report

**Audit Target:** `processed_v3`  
**Dataset Build Status:** `PASSED` (Build seed: `42`)  
**Audit Date:** October 17, 2026  
**Auditor:** Antigravity Independent Audit Subagent  

---

## 1. Executive Summary

This report delivers a thorough validation of the **structural validity axis** within the `processed_v3` dataset. The validity axis benchmarks a model's ability to recognize chemically impossible (valence-corrupted) molecular graphs versus valid molecules, completely isolated from semantic novelty.

The synthetic corruption pipeline introduces targeted, valence-violating edits grouped into three families:
1. **Family A (Bond-Order Inflation):** Clean chemical validity test; leaves all atom, bond, degree, and elemental counts unchanged.
2. **Family B (Bond Relocation):** Preserves atom and bond counts and graph connectivity, but shifts vertex degree distributions.
3. **Family C (Atom Swap — Held Out):** Replaces carbon atoms with oxygen atoms, testing generalization to unseen corruption types in test only.

All 2,172 corrupted graphs across train (1,086), validation (543), and test (543) partitions were analyzed against their exact in-distribution valid twins.

---

## 2. Corruption Family Architecture & Chemistry

```
+-----------------------------------------------------------------------------------------------+
|                                  SYNTHETIC CORRUPTION FAMILIES                                |
+-----------------------------------------------------------------------------------------------+
|  Family A: Bond Inflation     |  Family B: Bond Relocation     |  Family C: Atom Swap         |
|  - Edit: Single -> Double or  |  - Edit: Move one bond end     |  - Edit: Carbon -> Oxygen    |
|          Double -> Triple     |          to saturated atom     |          on distinct atoms   |
|  - Vertices (N): UNCHANGED    |  - Vertices (N): UNCHANGED     |  - Vertices (N): UNCHANGED   |
|  - Edges (E): UNCHANGED       |  - Edges (E): UNCHANGED        |  - Edges (E): UNCHANGED      |
|  - Degrees: UNCHANGED         |  - Degrees: SHIFTED (77-81%)   |  - Degrees: UNCHANGED        |
|  - Elements: UNCHANGED        |  - Elements: UNCHANGED         |  - Elements: SHIFTED (100%)  |
|  - Partitions: Train, Val, Tst|  - Partitions: Train, Val, Tst |  - Partitions: TEST ONLY     |
+-----------------------------------------------------------------------------------------------+
```

### Chemistry of Each Family

#### Family A (Bond-Order Inflation)
- **Mechanism:** Selects an existing non-aromatic single or double bond and inflates its bond order (single $\to$ double or double $\to$ triple).
- **Graph Invariance:** The unweighted adjacency structure and vertex degrees remain strictly identical. No elements are added or deleted.
- **Chemical Anomaly:** Induces explicit valence violations (e.g., pentavalent carbons or tetravalent nitrogens) solely through edge attribute modifications (`edge_attr[:, 0:4]`).
- **Benchmark Role:** Serves as the purest validity test because count-only models have exactly zero statistical signal to separate corrupted graphs from valid twins.

#### Family B (Bond Relocation)
- **Mechanism:** Takes one end of an existing single bond and relocates it to an already-saturated atom.
- **Graph Invariance:** Atom count ($N$) and edge count ($E$) are preserved. A graph connectivity check ensures no disconnected components are introduced. An upper bound ensures no degree-5+ vertices are created.
- **Chemical Anomaly:** The receiving saturated atom exceeds its standard valence, while the donor atom loses a bond.
- **Benchmark Role:** Introduces subtle topological distortion. Because the donor loses degree while the recipient gains degree, vertex degree histograms shift.

#### Family C (Carbon $\to$ Oxygen Atom Swap)
- **Mechanism:** Replaces a saturated or unsaturated carbon atom with an oxygen atom without adjusting attached bond orders or implicit hydrogens.
- **Chemical Anomaly:** Creates hypervalent oxygen atoms (e.g., trivalent or tetravalent oxygen) violating oxygen's standard octet valence of 2.
- **Benchmark Role:** Held-out corruption family evaluated **only on the test split** ($n=181$). Evaluates whether a model has learned a generalizable concept of physical validity or merely memorized the specific perturbation mechanisms seen during training.

---

## 3. Split Distribution & Edit Statistics

### Quota vs. Actual Counts
| Split | Family | Quota | Achieved | Status |
| :--- | :--- | :--- | :--- | :--- |
| `corrupt_train` | Family A | 543 | 543 | **PASS (100%)** |
| `corrupt_train` | Family B | 543 | 543 | **PASS (100%)** |
| `corrupt_val` | Family A | 272 | 272 | **PASS (100%)** |
| `corrupt_val` | Family B | 271 | 271 | **PASS (100%)** |
| `corrupt_test` | Family A | 181 | 181 | **PASS (100%)** |
| `corrupt_test` | Family B | 181 | 181 | **PASS (100%)** |
| `corrupt_test` | Family C | 181 | 181 | **PASS (100%)** |
| **Total** | | **2,172** | **2,172** | **PASS (100%)** |

### Edit Count Distributions
All corruptions consist of 1, 2, or 3 edits applied to distinct atoms:

| Split | Family | 1 Edit ($n$) | 2 Edits ($n$) | 3 Edits ($n$) | Mean Edits | Mean Violating Atoms |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `corrupt_train` | A | 266 (49.0%) | 195 (35.9%) | 82 (15.1%) | 1.66 | 2.13 |
| `corrupt_train` | B | 271 (49.9%) | 186 (34.3%) | 86 (15.8%) | 1.66 | 1.66 |
| `corrupt_val` | A | 118 (43.4%) | 94 (34.6%) | 60 (22.1%) | 1.79 | 2.33 |
| `corrupt_val` | B | 115 (42.4%) | 101 (37.3%) | 55 (20.3%) | 1.78 | 1.78 |
| `corrupt_test` | A | 76 (42.0%) | 65 (35.9%) | 40 (22.1%) | 1.80 | 2.20 |
| `corrupt_test` | B | 84 (46.4%) | 61 (33.7%) | 36 (19.9%) | 1.73 | 1.73 |
| `corrupt_test` | C | 72 (39.8%) | 66 (36.5%) | 43 (23.8%) | 1.84 | 1.84 |

**Observations:**
- In Family A, inflation of a single bond into a double bond creates valence violations on both flanking atoms, resulting in a mean violating atom count ($2.13 - 2.33$) higher than the raw edit count ($1.66 - 1.80$).
- In Family B, bond relocation specifically violates the recipient atom, matching the edit count ($1.66 - 1.78$).
- In Family C, each carbon-to-oxygen replacement directly creates an invalid oxygen center, matching the edit count ($1.84$).

---

## 4. Pair-Level Invariance & Shortcut Audit

To rigorously quantify whether corrupted molecules can be trivially detected by summary statistics, we computed twin-difference deltas across all 2,172 pairs:
$$\Delta(G_{\text{corrupt}}, G_{\text{twin}}) = \text{Feature}(G_{\text{corrupt}}) - \text{Feature}(G_{\text{twin}})$$

| Split / Family | Total Pairs | $\Delta \text{Atom Count} \ne 0$ | $\Delta \text{Edge Count} \ne 0$ | $\Delta \text{Degree Hist} \ne 0$ | $\Delta \text{Element Counts} \ne 0$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **train / Family A** | 543 | **0.0%** (0) | **0.0%** (0) | **0.0%** (0) | **0.0%** (0) |
| **train / Family B** | 543 | **0.0%** (0) | **0.0%** (0) | **81.2%** (441) | **0.0%** (0) |
| **val / Family A** | 272 | **0.0%** (0) | **0.0%** (0) | **0.0%** (0) | **0.0%** (0) |
| **val / Family B** | 271 | **0.0%** (0) | **0.0%** (0) | **78.6%** (213) | **0.0%** (0) |
| **test / Family A** | 181 | **0.0%** (0) | **0.0%** (0) | **0.0%** (0) | **0.0%** (0) |
| **test / Family B** | 181 | **0.0%** (0) | **0.0%** (0) | **76.8%** (139) | **0.0%** (0) |
| **test / Family C** | 181 | **0.0%** (0) | **0.0%** (0) | **0.0%** (0) | **100.0%** (181) |

### Interpretation of Shortcut Cues
1. **Family A is perfectly invariant:** No atom counts, edge counts, degrees, or element counts change. Any model that detects Family A must process edge features and understand valence rules.
2. **Family B has an unavoidable degree cue:** Over 76% of pairs have an altered degree histogram. For example, moving a bond from a CH2 to a CH creates a degree-1 and degree-4 vertex where degree-2 and degree-3 vertices were before. A count model can exploit this to achieve an AUROC of $0.65 - 0.74$.
3. **Family C has an explicit elemental composition cue:** Swapping carbon for oxygen decreases total carbon and increases total oxygen by 1 to 3 atoms. A simple linear classifier inspecting oxygen count achieves AUROC of $\approx 0.80$.

---

## 5. Twin Integrity & Split Partition Verification

Every corrupted molecule was cross-referenced with `master_table.csv` and `eligible_pairs.csv`:
- **Twin Split Alignment:** 100% of twins reside in the exact same split partition (`id_train` $\to$ `corrupt_train`, `id_val` $\to$ `corrupt_val`, `id_test` $\to$ `corrupt_test`).
- **Graph Size Alignment:** Every corrupted graph has identical heavy atom count to its valid twin ($N_{\text{corrupt}} \equiv N_{\text{twin}}$) and identical bond count ($E_{\text{corrupt}} \equiv E_{\text{twin}}$).
- **Label Integrity:** Corrupted graphs do not possess synthetic Tox21 labels; all 12 label entries are set to `NaN`, preventing inadvertent contamination of supervised multitask training.

---

## 6. Recommendations for Downstream GNN Benchmarking

1. **Always report validity AUROCs per family:** Grouping Families A, B, and C into a single "validity AUROC" obscures model failure modes. A model might achieve an apparent 0.80 AUROC simply by detecting oxygen counts in Family C and degree shifts in Family B, while completely failing on Family A.
2. **Benchmark Family A as the gold standard for valence reasoning:** Because Family A has a count-only AUROC of $\approx 0.50$, any performance above random chance directly reflects graph neural network relational reasoning.
3. **Include a count-only baseline:** Downstream publications should always report a count-only tabular baseline (e.g., Random Forest or Gradient Boosted Trees on degree/element counts) alongside GNN architectures to demonstrate how much predictive signal exceeds summary statistics.
