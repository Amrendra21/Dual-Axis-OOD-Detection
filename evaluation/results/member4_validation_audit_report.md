# Member 4 Validation Audit & Scientific Evaluation Report
**Role:** Member 5 (Evaluation, Benchmarking & Manuscript Lead)  
**Date:** 2026-10-21  
**Repository:** `Dual-Axis-OOD-Detection-member5`  
**Artifact Scope:** Validation Evaluation Reproduction, Matched-Control Analysis, and Dual-Axis Architectural Assessment

---

## 1. Executive Summary & Exact Metric Reproduction

The four validation metric records recorded in [`evaluation/results/validation_evaluation_summary.csv`](file:///c:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/evaluation/results/validation_evaluation_summary.csv) were audited and recomputed from the primary prediction source files:
- [`Member4_Prediction_Handoff/validity_validation_predictions.csv`](file:///c:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/Member4_Prediction_Handoff/validity_validation_predictions.csv)
- [`Member4_Prediction_Handoff/novelty_scaffold_validation_predictions.csv`](file:///c:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/Member4_Prediction_Handoff/novelty_scaffold_validation_predictions.csv)
- [`Member4_Prediction_Handoff/novelty_dataset_validation_predictions.csv`](file:///c:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/Member4_Prediction_Handoff/novelty_dataset_validation_predictions.csv)

All metrics were computed strictly through [`evaluation/metrics.py`](file:///c:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/evaluation/metrics.py) using `compute_all_metrics(y_true, y_score)`. The reproduced values match the existing summary to within machine precision ($< 10^{-16}$). The existing summary file was verified intact and left un-overwritten.

### Reproduced Validation Metrics Table

| Model Identifier | Axis | Benchmark / Partition | Score Column | AUROC | AUPR-Out | AUPR-In | FPR95 | FPR95 (%) | $N_{\text{ID}}$ | $N_{\text{OOD}}$ | $N_{\text{Total}}$ |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `validity_gine` | Axis 1 | `validity_family_a_val` | `validity_score` | **0.9966** | 0.9933 | 0.9985 | **0.0103** | 1.03% | 679 | 272 | 951 |
| `validity_gine` | Axis 1 | `validity_family_b_val` | `validity_score` | **0.9952** | 0.9934 | 0.9971 | **0.0044** | 0.44% | 679 | 271 | 950 |
| `novelty_mahalanobis` | Axis 2 | `novelty_scaffold_val` | `novelty_score` | **0.4566** | 0.3440 | 0.6030 | **0.9632** | 96.32% | 679 | 408 | 1,087 |
| `novelty_mahalanobis` | Axis 2 | `novelty_dataset_val` | `novelty_score` | **0.6074** | 0.3347 | 0.8111 | **0.9116** | 91.16% | 679 | 225 | 904 |

---

## 2. Evaluation Protocol & Statistical Verification

1. **Positive-Class Convention:**
   - In accordance with project protocol, ground truth label `0` strictly represents In-Distribution (ID, negative class) and label `1` represents Out-of-Distribution (OOD, positive class).
   - In `evaluation/metrics.py`:
     - $\text{AUROC}$ measures the probability that an OOD graph receives a higher score than an ID graph.
     - $\text{AUPR}_{\text{out}}$ treats $\text{OOD} = 1$ as positive.
     - $\text{AUPR}_{\text{in}}$ treats $\text{ID} = 0$ as positive, evaluated by inverting scores ($-S$).
     - $\text{FPR95}$ is computed by sorting anomaly scores descending and finding the false positive rate on ID samples at the threshold achieving $\ge 95\%$ true positive rate on OOD samples.

2. **Score Directions:**
   - `validity_score`: Sigmoid probability output $\in [0.0, 1.0]$. Higher values indicate higher probability of corruption / chemical invalidity.
   - `novelty_score`: Regularized Mahalanobis distance $D_M(z) \in [0.0, \infty)$ from the centroid of `id_train` embeddings. Higher values indicate greater distance from the training manifold.

3. **Tie Handling & UID Alignment:**
   - Continuous novelty scores exhibit **0 ties** across all validation samples (1,087 unique in scaffold val; 904 unique in dataset val).
   - Validity scores exhibit expected saturation ties near boundary limits (49 ties in Family A val; 78 in Family B val). `roc_curve` and `roc_auc_score` apply standard mid-rank / trapezoidal tie handling.
   - UID alignment confirms exactly 679 unique `id_val` molecules evaluated against the respective OOD sets with zero missing or duplicate UIDs per benchmark.

---

## 3. Scientific Analysis of Novelty Scaffold Validation AUROC (0.4566)

The validation AUROC for `novelty_scaffold_val` is **0.4566** (below random guessing of 0.50). 

### Distribution Analysis
Inspection of the score distributions reveals:
- **`id_val` ($N=679$):** Mean = 13.347, Median = 12.004, Std = 6.474, Max = 71.856
- **`novel_scaffold_val` ($N=408$):** Mean = 12.393, Median = 11.390, Std = 4.881, Max = 44.586

### Mechanistic Cause (Not a Score Direction Inversion)
1. **Scaffold-Disjoint Splitting:** In Tox21 dataset construction, `id_val` was generated to be scaffold-disjoint from `id_train`. Because the Mahalanobis estimator `estimator.npz` was fitted **exclusively on `id_train`** ($N=5,432$), the molecules in `id_val` possess chemical scaffolds never observed during training.
2. **Representation Dispersion:** The unseen scaffolds in `id_val` disperse their graph embeddings far into the periphery of the training embedding distribution, resulting in higher average Mahalanobis distances (13.35) than the curated `novel_scaffold_val` set (12.39).
3. **Integrity Rule:** Inverting the score post hoc is strictly rejected because:
   - On `novel_dataset_val`, the identical metric and score direction yields AUROC $0.6074$ ($> 0.50$).
   - On the test set (`novel_scaffold_test`), mean OOD score (13.18) exceeds mean ID score (12.98).
   - The sub-0.50 validation performance is an empirical reflection of the distribution shift between `id_train` and `id_val`.

---

## 4. Pre-Specified Matched-Control Analysis (Task NM)

Using the authoritative [`processed_v3_addendum/.../tables/matched_controls_filtered.csv`](file:///c:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/processed_v3_addendum/processed_v3_addendum/tables/matched_controls_filtered.csv), pre-specified matched-control evaluation was executed for all available validation pairs.

All matched pairs have identical ring status, $|\Delta\text{atoms}| \le 2$, and $|\Delta\text{bonds}| \le 2$, with all exclusions applied. Every pair joined with 100% coverage to the Member 4 validation files.

| Novelty Benchmark | Matched Pairs ($N$) | Unmatched (Raw) AUROC | Matched-Control AUROC (Task NM) | Matched AUPR | Matched FPR95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `novel_scaffold_val` vs matched ID controls | 372 | 0.4566 | **0.4783** | 0.4956 | 0.9866 |
| `novel_dataset_val` vs matched ID controls | 209 | 0.6074 | **0.5378** | 0.5296 | 0.9522 |

### Key Findings
1. **Scaffold Novelty:** Controlling for molecular size and ring presence slightly raises scaffold AUROC from $0.4566$ to $0.4783$, but it remains sub-0.50. This confirms that the validation deficit is structural and scaffold-driven rather than an artifact of molecular size imbalance.
2. **Dataset Novelty (SIDER):** Controlling for molecular size substantially reduces dataset AUROC from $0.6074$ down to $0.5378$ (near chance). This demonstrates that a large fraction of the raw discriminative capacity on SIDER was confounded by heavy atom count differences (SIDER molecules being systematically larger).

---

## 5. Dual-Axis Architectural Limitations & Missing Task D Inputs

### Independent Components vs. Unified Dual-Axis Architecture
The received Member 4 files do not constitute an integrated dual-axis model. They are two decoupled, independently executed algorithms:
1. `validity_gine`: A dedicated GINE binary classifier trained with outlier exposure on corrupted graphs (Families A & B).
2. `novelty_mahalanobis`: A post-hoc regularized Gaussian density estimator fitted on penultimate embeddings of a toxicity prediction network.

### Blocking Deficits for Joint Task D Evaluation
Task D evaluates a joint 3-class classification rule (In-Distribution vs. Chemically Invalid vs. Structurally Novel) using 95th-percentile thresholds $(\tau_v, \tau_n)$ derived from `id_val`.

Evaluating Task D is **scientifically impossible** from the supplied handoff CSVs due to missing data:
1. **Missing Cross-Axis Validity Scores:** [`validity_validation_predictions.csv`](file:///c:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/Member4_Prediction_Handoff/validity_validation_predictions.csv) contains scores only for `id_val` and `corrupt_val`. It lacks `validity_score` for `novel_scaffold_val` and `novel_dataset_val`. We cannot test whether novel molecules are falsely flagged as invalid.
2. **Missing Cross-Axis Novelty Scores:** [`novelty_*_validation_predictions.csv`](file:///c:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/Member4_Prediction_Handoff/) contain scores only for `id_val` and `novel_*_val`. They lack `novelty_score` for `corrupt_val`. We cannot test whether corrupted molecules are falsely classified as novel.
3. **No Fabricated Cross-Pairing:** Aligning scores across different molecules or imputing missing cross-axis predictions would constitute scientific fabrication and is strictly prohibited.
4. **No Claim of Dual-Axis Superiority:** Because Task D and cross-cause separation (Task X) cannot be computed on these files, no claim of dual-axis superiority over single-score baselines can be justified from Member 4's current handoff artifacts.

---

## 6. Distinguishing Empirical Observations from Hypotheses

- **Observed Empirical Facts:**
  1. `validity_gine` achieves near-perfect discrimination on validation corruption Families A (AUROC 0.9966) and B (AUROC 0.9952).
  2. `novelty_mahalanobis` achieves AUROC 0.4566 on scaffold novelty and 0.6074 on dataset novelty.
  3. Size-matched control matching attenuates dataset novelty AUROC from 0.6074 to 0.5378, proving strong size-confounding in SIDER.
  4. Cross-axis prediction scores are absent, preventing joint decision metrics.
- **Open Hypotheses (Requiring Unified Joint Models such as DAG-NN):**
  1. *Hypothesis 1:* A joint dual-axis model (DAG-NN) trained with dual objectives can suppress scaffold sensitivity on ID validation while maintaining high valence sensitivity.
  2. *Hypothesis 2:* True dual-axis separation (Task X) will outperform single-score energy and Mahalanobis baselines when evaluated on instances possessing both scores simultaneously.
