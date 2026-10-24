# DAG-NN Track A Mahalanobis Novelty Failure: Read-Only Forensic Diagnostic Report

**Lead Investigator:** Member 5 — Evaluation, Benchmarking & Manuscript Lead  
**Date:** 2026-10-21  
**Repository:** `C:\Users\Lenovo\Dual-Axis-OOD-Detection-member5`  
**Evaluation Scope:** Track A Candidate (`DAGNN`), 5 Seeds (0..4), Validation Partitions Only  
**Audit Protocol:** Strictly Read-Only; Frozen Baseline; Zero Test Partition Access  

---

## 1. Executive Summary & Diagnostic Verifications

This forensic diagnostic investigates why DAG-NN Track A's regularized Mahalanobis novelty detector produces sub-0.50 AUROC across all five validation seeds:
- **Scaffold Novelty AUROC:** $0.4539 \pm 0.0055$ (Task N, raw vs `id_val`)
- **Dataset Novelty AUROC:** $0.4521 \pm 0.0042$ (Task N, raw vs `id_val`)

### Verification Matrix

| Diagnostic Area | Status | Key Evidence & Citation |
| :--- | :---: | :--- |
| **Implementation** | **PASS** | `models/dual_axis_gnn.py:L152-172`, `L412-426`, `L446-450`. Layer 3 mean-pooled $z_G \in \mathbb{R}^{128}$; fitted strictly on 5,432 `id_train` graphs; score is $(z - \mu)^T P (z - \mu)$. Recomputing scores matches saved predictions to $< 10^{-5}$. |
| **Score Orientation** | **PASS** | `evaluation/metrics.py:L10-14`, `L161-182`. Convention is strictly higher score = larger distance = more OOD. Reversing the score is mathematically rejected: on matched dataset controls, AUROC is $> 0.50$ ($0.5488$), and reversing raw score yields $0.546$, which collapses matched dataset AUROC to $0.4512$. |
| **UID Alignment** | **PASS** | `processed_v3_addendum/.../tables/dual_axis_labels.csv`. All 1,855 validation predictions join with 100% UID coverage. Authoritative exclusions (`exclude == 0`) filter exactly 2 samples, yielding 678 `id_val`, 408 `novel_scaffold_val`, 224 `novel_dataset_val`, and 543 `corrupt_val`. |
| **Covariance / Numerical** | **PASS** | Precision matrix is $128 \times 128$, symmetric, finite, with condition number $\kappa = 4,071 \pm 510$. All eigenvalues strictly positive ($\lambda_{\min} \approx 0.042$, $\lambda_{\max} \approx 169.7$). Alternative distance metrics (Euclidean, Norm, Cosine) produce even worse AUROCs ($0.35 - 0.43$). |
| **Data Boundary & Splitting** | **FAIL** | **Confounder 1 (Graph Size / Mean Pooling):** Global mean pooling induces strong negative correlation between molecular size and novelty score ($r = -0.399$ on `id_val`). Small ID molecules (3–13 atoms) have mean novelty score $229.4$, while SIDER molecules (mean $30.2$ atoms) concentrate near $\mu$ (mean score $91.5$). When size-matched (Task NM), dataset AUROC swings $+0.0967$ to $0.5488$.<br>**Confounder 2 (Scaffold Split on `id_val`):** `id_val` is scaffold-disjoint from `id_train` ($73.4\%$ unseen scaffolds), placing `id_val` into the representation periphery (mean distance to $\mu = 7.63$) further than `novel_scaffold_val` ($6.75$). |

---

## 2. Comprehensive Per-Seed & Aggregate Diagnostic Findings

### Table 2.1: Authoritative Novelty Benchmark Results Across 5 Seeds

| Seed | Scaffold Raw AUROC | Scaffold Matched AUROC (Task NM) | Scaffold Reversed AUROC | Dataset Raw AUROC | Dataset Matched AUROC (Task NM) | Dataset Reversed AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 0.4491 | 0.4828 | 0.5509 | 0.4561 | 0.5477 | 0.5439 |
| **1** | 0.4486 | 0.4796 | 0.5514 | 0.4504 | 0.5440 | 0.5496 |
| **2** | 0.4507 | 0.4811 | 0.5493 | 0.4563 | 0.5566 | 0.5437 |
| **3** | 0.4613 | 0.4894 | 0.5387 | 0.4530 | 0.5485 | 0.5470 |
| **4** | 0.4598 | 0.4890 | 0.5402 | 0.4449 | 0.5471 | 0.5551 |
| **Mean $\pm$ Std** | **0.4539 $\pm$ 0.0055** | **0.4844 $\pm$ 0.0041** | **0.5461 $\pm$ 0.0055** | **0.4521 $\pm$ 0.0042** | **0.5488 $\pm$ 0.0042** | **0.5479 $\pm$ 0.0042** |

*Note: FPR95 on raw scaffold is $0.9378 \pm 0.0070$; on raw dataset is $0.9324 \pm 0.0111$. Matched scaffold FPR95 is $0.9457 \pm 0.0134$; matched dataset FPR95 is $0.9091 \pm 0.0162$.*

### Table 2.2: Mahalanobis Novelty Score Distribution Summaries (Across Seeds)

All values denote the squared Mahalanobis distance $D_M^2(z) = (z - \mu)^T P (z - \mu)$:

| Validation Partition | $N$ | Mean Score | Median Score | Interquartile Range [Q25, Q75] | Full Observed Range [Min, Max] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `id_val` | 678 | $106.36 \pm 3.63$ | $62.45 \pm 3.44$ | $[39.80 \pm 1.76,\; 107.38 \pm 6.84]$ | $[13.99,\; 2023.26]$ |
| `novel_scaffold_val` | 408 | $82.77 \pm 5.16$ | $54.80 \pm 3.73$ | $[40.23 \pm 1.83,\; 81.63 \pm 5.48]$ | $[19.02,\; 1527.49]$ |
| `novel_dataset_val` | 224 | $92.04 \pm 2.87$ | $54.13 \pm 3.39$ | $[39.10 \pm 1.63,\; 84.24 \pm 2.92]$ | $[18.85,\; 2193.41]$ |
| `corrupt_val` (Fam A) | 272 | $139.37 \pm 7.49$ | $83.15 \pm 4.70$ | $[50.69 \pm 2.03,\; 161.92 \pm 6.94]$ | $[17.18,\; 1638.64]$ |
| `corrupt_val` (Fam B) | 271 | $111.01 \pm 5.79$ | $71.50 \pm 4.14$ | $[46.26 \pm 2.76,\; 126.60 \pm 6.81]$ | $[17.61,\; 824.64]$ |

**Key Finding:** Across all 5 seeds, `id_val` has both a higher median ($62.45$) and a higher mean ($106.36$) than both `novel_scaffold_val` (median $54.80$, mean $82.77$) and `novel_dataset_val` (median $54.13$, mean $92.04$). The distribution of `id_val` has an extreme long right tail extending past $2,000$.

### Table 2.3: Embedding Geometry & Representation Diagnostics

| Metric / Property | `id_train` | `id_val` | `novel_scaffold_val` | `novel_dataset_val` |
| :--- | :---: | :---: | :---: | :---: |
| **Embedding Dimension ($d$)** | 128 | 128 | 128 | 128 |
| **Finite Values Check** | 100% Finite (0 NaN/Inf) | 100% Finite (0 NaN/Inf) | 100% Finite (0 NaN/Inf) | 100% Finite (0 NaN/Inf) |
| **Mean L2 Norm $\|z\|_2$** | $14.03 \pm 0.81$ | $14.04 \pm 0.82$ | $13.42 \pm 0.77$ | $13.29 \pm 0.74$ |
| **Norm Std Dev** | $1.44 \pm 0.05$ | $1.62 \pm 0.05$ | $1.03 \pm 0.06$ | $1.29 \pm 0.05$ |
| **Mean Euclidean Dist to $\mu$ ($\|z - \mu\|_2$)** | $7.42 \pm 0.24$ | **$7.63 \pm 0.26$** | **$6.75 \pm 0.20$** | **$6.43 \pm 0.13$** |
| **Total Embedding Variance $\sum \text{Var}(z_j)$** | $61.42 \pm 4.31$ | **$63.35 \pm 4.79$** | **$42.61 \pm 2.62$** | **$42.40 \pm 2.38$** |

**Key Finding:** Novel graphs possess $\sim 33\%$ lower embedding variance ($42.5$ vs $63.4$) and are systematically closer to the training centroid $\mu$ than the in-distribution validation molecules ($6.75$ and $6.43$ vs $7.63$).

### Table 2.4: ID-Training Precision Matrix Properties

| Property | Value (Mean $\pm$ Std across Seeds) | Interpretation / Verification |
| :--- | :---: | :--- |
| **Dimension** | $128 \times 128$ | Exact match to pooled representation $z_G$. |
| **Symmetry ($\|P - P^T\|_F$)** | $< 10^{-6}$ | Perfectly symmetric across all seeds. |
| **Minimum Eigenvalue ($\lambda_{\min}$)** | $0.0424 \pm 0.0059$ | Strictly positive; positive definite. |
| **Maximum Eigenvalue ($\lambda_{\max}$)** | $169.74 \pm 4.27$ | Bounded, no exploding spectrum. |
| **Condition Number ($\kappa$)** | $4,071 \pm 510$ | Stable inversion via `torch.linalg.pinv`. |
| **Shrinkage Regime** | Fixed linear: $\alpha = 0.01$ | $\Sigma_{\text{reg}} = 0.99 \Sigma + 0.01 \frac{\text{Tr}(\Sigma)}{d} I$ |

### Table 2.5: Alternative Distance Scoring Ablation (AUROC)

| Metric Evaluated | Scaffold Novelty AUROC | Dataset Novelty AUROC | Conclusion |
| :--- | :---: | :---: | :--- |
| **Mahalanobis Distance (Baseline)** | **$0.4539 \pm 0.0055$** | **$0.4521 \pm 0.0042$** | Saved model baseline. |
| **Euclidean Distance ($\|z - \mu\|_2$)** | $0.4146 \pm 0.0049$ | $0.3665 \pm 0.0073$ | **Worse than Mahalanobis.** |
| **Embedding L2 Norm ($\|z\|_2$)** | $0.4051 \pm 0.0073$ | $0.3525 \pm 0.0228$ | **Worse than Mahalanobis.** |
| **Cosine Distance ($1 - \cos(z, \mu)$)** | $0.4260 \pm 0.0079$ | $0.3761 \pm 0.0070$ | **Worse than Mahalanobis.** |

**Key Finding:** The inverted ranking is not created by the covariance matrix. Without the covariance matrix (using raw Euclidean distance), AUROC plummets to $0.41$ (scaffold) and $0.36$ (dataset). The covariance metric actually partially rescues the geometry.

### Table 2.6: Correlations

- **Embedding Norm vs. Mahalanobis Novelty Score (All Val):** $r = +0.5869 \pm 0.0474$
- **Embedding Norm vs. Mahalanobis Novelty Score (`id_val` only):** $r = +0.5781 \pm 0.0441$
- **Validity Score (Axis 1) vs. Novelty Score (Axis 2):** $r = +0.4423 \pm 0.0413$
- **Molecular Size (Atom Count) vs. Novelty Score (`id_val`):** **$r = -0.3993$** (negative correlation)

---

## 3. Comparison with Member 4 Baseline

### Table 3.1: Architectural and Methodological Comparison

| Factor | Member 4 (`novelty_mahalanobis`) | DAG-NN Track A Candidate | Diagnostic Impact |
| :--- | :--- | :--- | :--- |
| **Encoder Architecture** | `GINEToxicityModel` (4 layers, BatchNorm) | `DAGNN` (3 layers, LayerNorm) | Capacity matched to S1/S2 baselines. |
| **Training Objective** | Supervised Tox21 classification only | Multi-task: Tox21 + node/edge masking | Masked loss regularizes representations. |
| **Embedding Layer** | Penultimate layer (before classifier) | Layer 3 global pooled representation | Representation space $z_G$. |
| **Graph Pooling** | **`mean_sum`** ($\text{mean} \parallel \text{sum}$) | **`mean`** only | **Critical Driver of SIDER Discrepancy.** |
| **Embedding Dimension** | $256$ ($128 \times 2$) | $128$ | Member 4 includes sum pooling. |
| **Covariance Shrinkage** | Scikit-learn Ledoit-Wolf ($\alpha \approx 0.00378$) | Fixed linear shrinkage ($\alpha = 0.01$) | Both well-conditioned. |
| **Distance Metric** | $D_M(z) = \sqrt{(z - \mu)^T P (z - \mu)}$ | $D_M^2(z) = (z - \mu)^T P (z - \mu)$ | Monotonic transformation (AUROC invariant). |

### Table 3.2: Metric Reproduction & Exclusion Verification

| Benchmark Partition | Member 4 Raw (Audit) | Member 4 Filtered (with Exclusions) | DAG-NN Track A (5-seed Mean) |
| :--- | :---: | :---: | :---: |
| `novelty_scaffold_val` (Raw vs `id_val`) | $0.4566$ | **$0.4570$** | **$0.4539 \pm 0.0055$** |
| `novelty_scaffold_matched_controls` (Task NM) | $0.4783$ | **$0.4783$** | **$0.4844 \pm 0.0041$** |
| `novelty_dataset_val` (Raw vs `id_val`) | $0.6074$ | **$0.6104$** | **$0.4521 \pm 0.0042$** |
| `novelty_dataset_matched_controls` (Task NM) | $0.5378$ | **$0.5378$** | **$0.5488 \pm 0.0042$** |

### Insights from the Comparison
1. **Verification of Member 4:** Applying the authoritative exclusion policy (`exclude == 0`) to Member 4's prediction files yields exactly **$0.6104$** on dataset novelty and **$0.4570$** on scaffold novelty.
2. **Scaffold Novelty Agreement:** Both Member 4 ($0.4570$) and DAG-NN ($0.4539$) fail on raw scaffold novelty in the exact same range ($\sim 0.45$). When size-matched (Task NM), both rise to $\sim 0.48$. This demonstrates that the sub-0.50 scaffold failure is **not unique to DAG-NN**; it is an inherent property of the benchmark's scaffold-disjoint validation split.
3. **The SIDER Pooling Divergence:** Member 4 achieved $0.6104$ on raw dataset novelty because its embedding included **sum pooling** (`h_sum`), which directly scaled with heavy atom count. SIDER molecules have an average of $30.2$ atoms vs $16.8$ for `id_train`. This size disparity pushed SIDER embeddings far from the center, creating an artificial AUROC of $0.6104$. When Member 4 controlled for size in Task NM, AUROC collapsed to $0.5378$. In DAG-NN, where **mean pooling** was used, larger molecules average out node vectors and concentrate *closer* to $\mu$, inverting raw dataset AUROC to $0.4521$. When DAG-NN controlled for size in Task NM, AUROC recovered to $0.5488$.

---

## 4. Ranked List of Evidence-Supported Explanations

### Rank 1: Confounder of Molecular Size under Global Mean Pooling (Proven)
- **Mechanism:** Global mean pooling $z_G = \frac{1}{N} \sum_{i=1}^N h_i$ has sample variance proportional to $1/N$. Small molecules have volatile embeddings with extreme distances from the centroid $\mu$, while large molecules average out variation and collapse toward $\mu$.
- **Measurements:**
  - Grouping all validation molecules into size quintiles reveals that small molecules ($3 - 13$ atoms) have a mean novelty score of **$229.39$**, while large molecules ($31 - 100$ atoms) have a mean novelty score of **$52.60$**.
  - Negative correlation between atom count and novelty score: $r = -0.3993$ on `id_val`.
  - SIDER molecules are systematically larger (mean $30.2$ atoms) than `id_train` ($16.8$ atoms) and `id_val` ($22.5$ atoms), forcing their mean-pooled embeddings into the interior of the distribution (median score $54.1$ vs $62.5$ for `id_val`).
  - Controlling for size and ring status (Task NM) produces an immediate $+0.0967$ AUROC jump (from $0.4521$ to $0.5488$).

### Rank 2: Scaffold-Disjoint Splitting of `id_val` Relative to `id_train` (Proven)
- **Mechanism:** Tox21 `id_val` is split by scaffold from `id_train`. Out of 511 unique scaffolds in `id_val`, **375 scaffolds ($73.4\%$) are completely unseen in `id_train`**.
- **Measurements:**
  - Because the Gaussian centroid $\mu$ and precision matrix $P$ are fitted exclusively on `id_train`, `id_val` molecules are themselves structurally novel relative to the reference distribution.
  - `id_val` molecules disperse far into the periphery (mean Euclidean distance to $\mu = 7.63$, total variance $63.35$), exceeding the dispersion of `novel_scaffold_val` (mean distance $6.75$, total variance $42.61$).
  - Evaluating `novel_scaffold_val` against `id_val` is comparing two different sets of out-of-training scaffolds, rather than comparing OOD against true in-distribution scaffolds.

### Rank 3: Embedding Compression under Multi-Task Regularization (Supported)
- **Mechanism:** DAG-NN trains simultaneously on Tox21 assay labels and masked feature reconstruction on `id_train`. The reconstruction loss encourages the model to learn prototypical, valid valence configurations.
- **Measurements:**
  - Chemically intact novel molecules (whether scaffold or dataset) conform to valid chemical rules and thus produce smooth, prototypical node representations that project near the centroid $\mu$.
  - In contrast, small or peripheral ID molecules deviate further from the average representation.
  - This explains why novel molecules exhibit significantly lower total variance ($42.5$) than ID molecules ($63.4$).

### Rank 4: Covariance Estimation or Matrix Conditioning (Disproven as Cause)
- **Measurements:** Precision matrices are symmetric, strictly positive-definite ($\lambda_{\min} \approx 0.042$), well-conditioned ($\kappa \approx 4,071$), with finite elements. Furthermore, removing the covariance matrix entirely and scoring via Euclidean distance makes the AUROC significantly worse ($0.36 - 0.41$). Covariance estimation is not the source of failure.

### Rank 5: Implementation or Sign Orientation Error (Disproven as Cause)
- **Measurements:** Score orientation adheres to the universal standard (higher = more OOD). Inverting the score post hoc is rejected: on matched dataset controls, AUROC is $0.5488$ (reversing would make it $0.4512$), and on raw dataset, reversed AUROC is only $0.5479$, which is statistically marginal and lacks scientific validity.

---

## 5. Explicit Limitations & Omitted Analyses

1. **Test Set Isolation:** Under no circumstances were test graphs, test predictions, or test checkpoints loaded, inspected, or scored. All findings are derived exclusively from training and validation partitions.
2. **Member 4 Checkpoint Re-Execution:** Member 4's raw `.pt` weights were not bundled in the handoff (`checkpoints/toxicity_gine/best_model.pt` was absent), preventing live forward-pass re-extraction of Member 4 embeddings. Member 4 analyses were performed on the official handoff prediction CSVs and audit records.
3. **No Retraining or Hyperparameter Tuning:** Baseline model weights and calibration parameters were held completely frozen throughout this diagnostic.

---

## 6. Single Most Informative Next Experiment

### Experiment Name: **Size-Normalized Graph Representation Calibration (Task N-Norm Validation Pilot)**

- **Objective:** Definitively isolate the degree to which graph-size bias under mean pooling causes the sub-0.50 AUROC, while keeping the trained DAG-NN backbone weights completely frozen.
- **Design (Strictly Validation-Only):**
  1. Freeze the existing trained checkpoint weights from Seed 0 (`checkpoints/dagnn_trackA/dagnn_trackA_seed0.pt`).
  2. Extract node representations $h_v \in \mathbb{R}^{128}$ for all graphs in `id_train`, `id_val`, `novel_scaffold_val`, and `novel_dataset_val`.
  3. Instead of unweighted mean pooling $z_G = \frac{1}{N} \sum h_v$, evaluate two size-stabilized representations:
     - **Option A (Size-Augmented Embedding):** $z_{\text{aug}} = [z_G \parallel \log(N)] \in \mathbb{R}^{129}$.
     - **Option B (Size-Stratified Mahalanobis):** Fit mean and covariance conditioned on molecular size bins ($N \le 15$, $16 \le N \le 25$, $N \ge 26$).
  4. Evaluate raw and matched AUROC on validation splits only.
- **Hypothesis Disconfirmation Criterion:**
  - If size normalization raises raw scaffold and dataset AUROC to $\ge 0.55$, the failure is confirmed to be an aggregation/pooling artifact rather than a failure of graph feature learning.
  - If size normalization fails to improve AUROC above $0.50$, the failure is confirmed to be driven by the scaffold-disjoint construction of `id_val`.
