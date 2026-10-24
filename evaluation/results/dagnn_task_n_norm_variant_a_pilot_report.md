# DAG-NN Task N-Norm Variant A Validation Pilot Report

**Lead Investigator:** Member 5 — Evaluation, Benchmarking & Manuscript Lead  
**Date:** 2026-10-21  
**Repository:** `C:\Users\Lenovo\Dual-Axis-OOD-Detection-member5`  
**Evaluation Scope:** Track A Candidate (`DAGNN`), Seed 0 Frozen Checkpoint (`checkpoints/dagnn_trackA/dagnn_trackA_seed0.pt`)  
**Pilot Target:** Variant A — Size-Augmented Mahalanobis Distance ($z_{\text{aug}} = [z_G \parallel \log(N)] \in \mathbb{R}^{129}$)  
**Audit Protocol:** Strictly Validation-Only; Frozen Backbone; Zero Test Partition Access  

---

## 1. Executive Summary & Baseline Reproduction

This exploratory pilot investigates whether augmenting DAG-NN's 128-dimensional graph embedding with explicit molecular size information ($\log(N)$) resolves the sub-0.50 Mahalanobis novelty failure observed on Track A validation benchmarks.

### Baseline Reproduction Status (Seed 0)
- **Checkpoint:** `checkpoints/dagnn_trackA/dagnn_trackA_seed0.pt` (frozen).
- **In-Memory Verification:** Recomputed novelty scores across all 1,310 non-excluded validation graphs match the saved prediction CSV (`predictions/DAGNN__trackA__seed0__val.csv`) with:
  - Maximum absolute difference: **$1.0986 \times 10^{-3}$**
  - Mean absolute difference: **$5.7071 \times 10^{-5}$**
  - Maximum relative difference: **$7.1810 \times 10^{-6}$** (well within float32 machine tolerance).
- **Alignment:** Graph embeddings, node counts, UIDs, and authoritative exclusion labels (`exclude == 0`) join with 100% coverage.
- **Reproduced Baseline Metrics (Seed 0):**
  - Scaffold Novelty Raw AUROC: **$0.4491$** (AUPR-Out: 0.3321, FPR@95: 0.9381)
  - Scaffold Novelty Matched AUROC (Task NM): **$0.4828$** (AUPR-Out: 0.4859, FPR@95: 0.9516)
  - Dataset Novelty Raw AUROC: **$0.4561$** (AUPR-Out: 0.2255, FPR@95: 0.9248)
  - Dataset Novelty Matched AUROC (Task NM): **$0.5477$** (AUPR-Out: 0.5255, FPR@95: 0.9139)

Baseline reproduction is **VERIFIED** and intact.

---

## 2. Variant A: Method & Covariance Configuration

### Representation Construction
For each graph $g$, we extract the frozen Layer 3 mean-pooled representation $z_G \in \mathbb{R}^{128}$ and the heavy atom count $N = |V(g)|$. The augmented representation is constructed as:
$$z_{\text{aug}} = [z_G \parallel \log(N)] \in \mathbb{R}^{129}$$

### Training Reference Distribution (id_train Only)
The reference mean $\mu_{\text{aug}} \in \mathbb{R}^{129}$ and regularized covariance $\Sigma_{\text{reg}} \in \mathbb{R}^{129 \times 129}$ were fitted strictly on the $5,432$ intact graphs of `id_train`:
- **Training Statistics of $\log(N)$:** Mean = $2.6894$, Std = $0.5118$, Min = $1.0986$ ($N=3$), Max = $4.5539$ ($N=95$).
- **Variance Balance:** Coordinate-wise average variance of $z_G$ in `id_train` is $0.4488$ ($\text{Tr}(\Sigma)/128 = 57.44 / 128$). The variance of $\log(N)$ is $0.5118^2 = 0.2619$, which naturally matches the magnitude of latent dimensions without artificial dominance.
- **Standardization Ablation:** Both raw $\log(N)$ and standardized $\log(N)$ ($z = \frac{\log(N) - 2.6894}{0.5118}$) were tested. Because shrinkage operates on trace-normalized identity, both configurations produced identical eigenvalues and identical metrics to within $0.0003$ AUROC.

### Covariance Regularization & Inversion
Following the project's established procedure in `models/dual_axis_gnn.py`:
$$\Sigma = \frac{1}{N_{\text{train}} - 1} \sum_{i=1}^{N_{\text{train}}} (z_{\text{aug}, i} - \mu_{\text{aug}})(z_{\text{aug}, i} - \mu_{\text{aug}})^T$$
$$\Sigma_{\text{reg}} = (1 - \alpha) \Sigma + \alpha \frac{\text{Tr}(\Sigma)}{d} I_{129}, \quad \text{with } \alpha = 0.01$$
$$P_{\text{aug}} = \text{pinv}(\Sigma_{\text{reg}})$$

### Covariance Numerical Health Diagnostics
- **Dimensions:** $129 \times 129$
- **Symmetry ($\|P - P^T\|_F$):** $< 10^{-6}$ (strictly symmetric)
- **Finite Check:** 100% finite (0 NaN, 0 Inf)
- **Minimum Eigenvalue ($\lambda_{\min}$):** $0.0495$ (strictly positive definite)
- **Maximum Eigenvalue ($\lambda_{\max}$):** $176.51$
- **Condition Number ($\kappa$):** $3,562$ (well-conditioned; stable inversion)
- **Regularization Warnings:** Zero. No jitter or ad-hoc diagonal loading required.

---

## 3. Validation-Only Benchmark Evaluation Results

All evaluations use the authoritative validation partitions (`exclude == 0`) and fixed score direction (larger distance = more OOD).

### Table 3.1: Metric Summary (Baseline vs. Variant A)

| Evaluation Benchmark | Partition / Subset | Metric | Baseline (128-dim) | Variant A (129-dim) | Difference ($\Delta = \text{Var} - \text{Base}$) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Scaffold Novelty (Raw)** | `novel_scaffold_val` vs `id_val`<br>($N_{\text{OOD}}=408, N_{\text{ID}}=678$) | **AUROC**<br>AUPR-Out<br>FPR@95 | **0.4491**<br>0.3321<br>0.9381 | **0.4454**<br>0.3308<br>0.9410 | **-0.0037**<br>-0.0013<br>+0.0029 |
| **Scaffold Novelty (Matched)** | `novel_scaffold_val` vs matched ID<br>(Task NM, $N_{\text{pairs}}=372$) | **AUROC**<br>AUPR-Out<br>FPR@95 | **0.4828**<br>0.4859<br>0.9516 | **0.4829**<br>0.4859<br>0.9435 | **+0.0001**<br>0.0000<br>-0.0081 |
| **Dataset Novelty (Raw)** | `novel_dataset_val` vs `id_val`<br>($N_{\text{OOD}}=224, N_{\text{ID}}=678$) | **AUROC**<br>AUPR-Out<br>FPR@95 | **0.4561**<br>0.2255<br>0.9248 | **0.4666**<br>0.2283<br>0.9248 | **+0.0105**<br>+0.0028<br>0.0000 |
| **Dataset Novelty (Matched)** | `novel_dataset_val` vs matched ID<br>(Task NM, $N_{\text{pairs}}=209$) | **AUROC**<br>AUPR-Out<br>FPR@95 | **0.5477**<br>0.5255<br>0.9139 | **0.5463**<br>0.5249<br>0.8947 | **-0.0014**<br>-0.0006<br>-0.0192 |

### Table 3.2: Score Distributions by Group (Squared Mahalanobis Distance)

| Model Variant | Partition | Mean $\pm$ Std | Median [Q25, Q75] | Range [Min, Max] |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline (128-dim)** | `id_val` ($N=678$) | $110.34 \pm 141.47$ | $66.05 \; [41.79, 112.60]$ | $[16.31,\; 1765.79]$ |
| | `novel_scaffold_val` ($N=408$) | $86.00 \pm 112.71$ | $56.57 \; [41.11, 86.65]$ | $[19.45,\; 1410.01]$ |
| | `novel_dataset_val` ($N=224$) | $94.16 \pm 155.26$ | $58.03 \; [41.34, 87.66]$ | $[21.89,\; 1895.83]$ |
| **Variant A (129-dim)** | `id_val` ($N=678$) | $112.27 \pm 142.52$ | $67.15 \; [43.01, 114.21]$ | $[20.00,\; 1787.62]$ |
| | `novel_scaffold_val` ($N=408$) | $87.37 \pm 113.49$ | $57.54 \; [42.42, 87.63]$ | $[20.01,\; 1411.42]$ |
| | `novel_dataset_val` ($N=224$) | $97.45 \pm 158.62$ | $63.55 \; [44.50, 88.87]$ | $[24.27,\; 1955.85]$ |

### Table 3.3: Node Count Correlations ($r(N, \text{Score})$)

| Split / Cohort | Baseline (128-dim) | Variant A (129-dim) | Correlation Shift ($\Delta r$) |
| :--- | :---: | :---: | :---: |
| `id_val` ($N=678$) | $-0.3993$ | $-0.3823$ | $+0.0170$ |
| `novel_scaffold_val` ($N=408$) | $-0.2854$ | $-0.2733$ | $+0.0121$ |
| `novel_dataset_val` ($N=224$) | $-0.2919$ | $-0.2616$ | $+0.0303$ |
| **All Validation Combined** ($N=1,310$) | **$-0.3389$** | **$-0.3189$** | **$+0.0200$** |

---

## 4. Key Findings & Diagnostic Inferences

1. **Inconsistent and Minimal Metric Impact:**
   - On **dataset novelty**, appending $\log(N)$ marginally increases raw AUROC by **$+0.0105$** (from $0.4561$ to $0.4666$).
   - On **scaffold novelty**, appending $\log(N)$ marginally decreases raw AUROC by **$-0.0037$** (from $0.4491$ to $0.4454$).
   - On **matched controls (Task NM)**, where molecular size is already strictly held constant ($|\Delta\text{atoms}| \le 2$), Variant A has zero effect on scaffold AUROC ($+0.0001$) and a negligible negative effect on dataset AUROC ($-0.0014$).
2. **Persistence of Negative Size Correlation:**
   - Appending $\log(N)$ only attenuated the negative correlation between molecular size and novelty score by $0.02$ (from $-0.3389$ to $-0.3189$).
   - **Mathematical Explanation:** Mean pooling causes the variance of the 128 latent node embeddings to contract as $\mathcal{O}(1/N)$ across all 128 dimensions. Adding a single 129th coordinate $\log(N)$ contributes only a single scalar quadratic term $(\log(N) - \mu_{\log N})^2 P_{129,129}$, which is mathematically overwhelmed by the remaining 128 contracted dimensions.
3. **Hypothesis Resolution:**
   - Simply concatenating $\log(N)$ as a linear feature in post-hoc Mahalanobis scoring is **insufficient to resolve the sub-0.50 novelty ranking**.
   - The sub-0.50 score is not caused by the absence of a size coordinate in the density model; it is driven by (a) non-linear embedding variance compression inherent to mean pooling across all hidden dimensions, and (b) the severe scaffold-disjoint divergence of `id_val` from `id_train`.

---

## 5. Limitations & Protocol Compliance

1. **Validation-Only Scope:** No test graphs, labels, predictions, or test checkpoints were loaded or inspected.
2. **Frozen Encoder:** The underlying neural network parameters and node embeddings were not retrained or fine-tuned.
3. **Single Seed Pilot:** Findings are exploratory and restricted to Seed 0; no claim of statistical significance across training random seeds is made.
4. **Baseline Unaltered:** The authoritative five-seed baseline remains completely intact and un-overwritten.

---

## 6. Recommendation for Next Experiment

Appending a single linear size feature to an unnormalized mean-pooled embedding failed because the 128 pooled dimensions still suffer from $1/N$ variance shrinkage.

The most informative next experiment is:
### **Variance-Stabilized Graph Aggregation (Task N-Pool Pilot)**
- **Design:** Keep the backbone frozen. Instead of standard unweighted mean pooling $z_G = \frac{1}{N} \sum h_v$, evaluate variance-stabilized pooling schemes that prevent graph-size contraction:
  1. **Normalized Sum Pooling:** $z_{\text{norm}} = \frac{1}{\sqrt{N}} \sum_{v=1}^N h_v$ (scales node variance uniformly according to the Central Limit Theorem).
  2. **Size-Conditioned Covariance:** Fit covariance $\Sigma_k$ conditioned on graph size tiers ($N \le 15$, $16 \le N \le 25$, $N \ge 26$) so that small ID molecules do not distort the reference distribution of larger molecules.
