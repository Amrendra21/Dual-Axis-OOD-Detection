# Paper Readiness Checklist: Dual-Axis OOD Detection in Molecular Graphs
**Date:** October 17, 2026  
**Auditor / Role:** Member 4 (Model Development & Training)  
**Target:** Publication Submission (e.g., NeurIPS, ICLR, ICML, JCIM)  
**Status:** Audit Completed / Ready for Paper Draft

---

## 1. Experimental Design & Protocol Integrity

- [x] **Strict Separation of Evaluation Axes:** Chemical validity detection (Axis 1) and structural novelty detection (Axis 2) are maintained as independent tasks with separate evaluation protocols.
- [x] **No Test Set Tuning / Model Selection:** Model architectures, hyperparameters, regularization shrinkage ($\alpha$), and baseline comparisons were selected strictly on validation data (`id_val`, `corrupt_val`, `novel_scaffold_val`, `novel_dataset_val`). Final test sets remain un-tuned.
- [x] **Family C Unseen Holdout:** Family C (valence corruptions) is strictly quarantined in `corrupt_test` ($n=181$) and completely absent from all training and validation splits.
- [x] **Cross-Split SMILES Leakage:** 0 cross-split molecular SMILES leakage confirmed across all 45 split pairs in `processed_v3/tables/master_table.csv`.
- [x] **UID Alignment:** All 10,140 graphs preserve unique, 1-to-1 string identifiers (`T21-*`, `COR-*`, `SID-*`) with zero index shifts between `.pt` graph files and metadata tables.

---

## 2. Benchmark Validity & Critical Findings

- [x] **Murcko Scaffold Disjointness Audited:**
  - `id_train` contains 850 unique Murcko scaffolds.
  - `id_val` contains 509 unique Murcko scaffolds.
  - Overlap between `id_train` and `id_val` is **0 (0.0%)**.
  - Overlap between `id_train` and `novel_scaffold_val` is **0 (0.0%)**.
  - Overlap between `id_val` and `novel_scaffold_val` is **0 (0.0%)**.
- [x] **Scientific Implication Documented:** Because `id_val` is already scaffold-disjoint from `id_train`, estimating structural novelty via distance to `id_train` penalizes both in-distribution validation negatives and novel-scaffold positives. Under size- and ring-matched controls, all methods (Mahalanobis, kNN, Energy, and ECFP4) perform around chance ($\text{AUROC} \approx 0.47 - 0.50$). This crucial finding must be transparently reported as a benchmark characteristic in the paper.
- [x] **Caliper Matched Controls:** Matched controls (`matched_controls.csv`) pair OOD instances with ID validation instances strictly within $\Delta \text{atoms} \le 2$ and identical ring status (`is_acyclic`), eliminating trivial size-confounding. Coverage: 91.18% (scaffold) and 93.78% (dataset).

---

## 3. Validity Detection Shortcut Analysis

- [x] **Simple Graph Statistics Baselines:** Atom count ($N$), bond count ($E$), and 32-element count vectors completely fail on Family A corruptions ($\text{AUROC} = 0.51 - 0.56$), whereas the GINE validity detector achieves **0.9966 AUROC** on Family A.
- [x] **Spurious Correlation Check:** Point-biserial correlations between `validity_score` and graph properties in validation are negligible and statistically non-significant:
  - $\text{corr}(\text{validity\_score}, N) = -0.0338$ ($p = 0.24$)
  - $\text{corr}(\text{validity\_score}, E) = -0.0321$ ($p = 0.26$)
  - $\text{corr}(\text{validity\_score}, \text{acyclic}) = -0.0162$ ($p = 0.57$)
- [x] **Conclusion:** The validity detector does not exploit graph size or element abundance shortcuts; it genuinely detects valence and connectivity anomalies.

---

## 4. Statistical Validation & Significance

- [x] **Bootstrap Confidence Intervals:** 95% bootstrap confidence intervals (1,000–2,000 resamples) computed for all primary metrics (AUROC, AUPR, FPR95).
- [x] **Paired Hypothesis Testing:** 2,000-replicate paired bootstrap hypothesis tests conducted for difference in AUROC ($\Delta \text{AUROC}$) across main novelty methods.
- [x] **Honest Significance Claims:** Under matched controls, neither ECFP4 nor Multi-Task Energy demonstrates statistically significant improvement over GNN Mahalanobis ($p > 0.40$, 95% CIs overlap 0.0). No false claims of superiority are made.

---

## 5. Software & Environment Reproducibility

- [x] **Independent Metric Recomputation:** 100% of reported Step 5 and Step 7 metrics independently recomputed and verified to bit-exact precision ($0.00\times 10^0$ error).
- [x] **Regression Test Suite:** Full 6-suite regression sequence (`test_loader.py`, `test_model.py`, `test_validity.py`, `test_novelty.py`, `test_integration.py`, `test_step7_novelty.py`) executed and passing with Exit Code 0.
- [x] **Environment Frozen:** Exact versions pinned and documented in `step8_validation/environment_versions.txt` (Python 3.10.0, PyTorch 2.11.0+cpu, RDKit 2026.09.1, scikit-learn 1.4.2, SciPy 1.15.3, NumPy 1.26.4, Pandas 2.2.2).
- [x] **Deterministic Seeding:** Master seed 42 enforced across PyTorch, NumPy, and Scikit-Learn sampling routines.

---

## 6. Paper Readiness Summary

| Component | Status | Readiness Level |
|:---|:---:|:---:|
| Axis 1: Validity Detection | Verified | **Ready for Publication** (>0.995 AUROC, non-shortcut verified) |
| Axis 2: Novelty Detection | Audited & Validated | **Ready for Publication** (Transparent benchmark report & baseline study) |
| Multi-Task Toxicity Backbone | Frozen & Verified | **Ready for Publication** (Macro ROC-AUC = 0.7423) |
| Statistical Rigor | Complete | **Ready for Publication** (Paired bootstrap CIs and p-values recorded) |
| Codebase & Tests | Passing | **Ready for Release** (6 / 6 test suites passing) |
