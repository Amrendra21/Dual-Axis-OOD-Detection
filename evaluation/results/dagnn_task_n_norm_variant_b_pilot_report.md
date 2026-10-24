# DAG-NN Task N-Norm Variant B Validation Pilot Report
## Normalized-Sum Pooling: $z_{\mathrm{norm}} = \frac{1}{\sqrt{N}}\sum_{v}h_v$

**Lead Investigator:** Member 5 — Evaluation, Benchmarking & Manuscript Lead  
**Date:** 2026-10-21  
**Repository:** `C:\Users\Lenovo\Dual-Axis-OOD-Detection-member5`  
**Checkpoint:** `checkpoints/dagnn_trackA/dagnn_trackA_seed0.pt` (frozen, unmodified)  
**Scope:** Validation-only; zero test-partition access; frozen backbone; no retraining  
**Prior Reports:**
- [`evaluation/results/dagnn_novelty_failure_diagnostic_report.md`](file:///C:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/evaluation/results/dagnn_novelty_failure_diagnostic_report.md)
- [`evaluation/results/dagnn_task_n_norm_variant_a_pilot_report.md`](file:///C:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/evaluation/results/dagnn_task_n_norm_variant_a_pilot_report.md)

---

## 1. Path & Baseline Pre-flight

**Output path check:** `evaluation/results/dagnn_task_n_norm_variant_b_pilot_report.md` did not exist before creation. **Proceeding.**

**Baseline Preservation:** Existing prediction CSVs, checkpoints, five-seed summaries, labels, split manifests, and prior reports are unmodified. The five-seed benchmark summary in `evaluation/results/dagnn_trackA_5seeds_benchmark_summary.csv` remains the authoritative baseline.

---

## 2. Baseline Reproduction Status (Seed 0)

- **Checkpoint:** `checkpoints/dagnn_trackA/dagnn_trackA_seed0.pt` loaded read-only.
- **Reproduction Source:** In-memory recomputation of squared Mahalanobis distance $(z_G - \mu)^T P (z_G - \mu)$ using the stored `novelty_mu` and `novelty_prec` buffers from the checkpoint state dict.
- **Comparison Target:** `predictions/DAGNN__trackA__seed0__val.csv`

| Reproduction Metric | Value |
| :--- | :---: |
| Maximum Absolute Difference | $1.0986 \times 10^{-3}$ |
| Mean Absolute Difference | $5.7071 \times 10^{-5}$ |
| Maximum Relative Difference | $7.1810 \times 10^{-6}$ |
| **Status** | **PASS** (within float32 machine precision) |

**Active Sample Counts After `exclude == 0` Filter:**

| Cohort | Expected | Observed | Match |
| :--- | :---: | :---: | :---: |
| `id_val` | 678 | **678** | ✓ |
| `novel_scaffold_val` | 408 | **408** | ✓ |
| `novel_dataset_val` | 224 | **224** | ✓ |
| Matched scaffold pairs (Task NM) | 372 | **372** | ✓ |
| Matched dataset pairs (Task NM) | 209 | **209** | ✓ |

**Reproduced Seed 0 Baseline Metrics:**
| Benchmark | AUROC | AUPR-Out | FPR@95 |
| :--- | :---: | :---: | :---: |
| Scaffold Novelty (Raw) | 0.4491 | 0.3321 | 0.9381 |
| Scaffold Novelty (Matched, Task NM) | 0.4828 | 0.4859 | 0.9516 |
| Dataset Novelty (Raw) | 0.4561 | 0.2255 | 0.9248 |
| Dataset Novelty (Matched, Task NM) | 0.5477 | 0.5255 | 0.9139 |

Baseline is **verified and stable**. Proceeding.

---

## 3. Normalized-Sum Representation: Methodology & Implementation

### 3.1 Representation Definition

For each graph $g$ with $N$ nodes, the normalized-sum representation is:

$$z_{\mathrm{norm}}(g) = \frac{1}{\sqrt{N}} \sum_{v=1}^{N} h_v \in \mathbb{R}^{128}$$

where $h_v \in \mathbb{R}^{128}$ are the **pre-pooling node representations** produced by the frozen 3-layer edge-aware GIN backbone. These are the same node representations $h$ output by `embed_nodes_and_graph` in [`models/dual_axis_gnn.py`](file:///C:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/models/dual_axis_gnn.py#L152-L172) before any pooling is applied. The existing mean-pooled output $z_G$ is not reused; $h$ is extracted directly from the identical forward pass.

The node count $N$ is defined as the heavy-atom count `g["x"].shape[0]`, **identical to the convention used in Variant A**. No model parameters or checkpoint state were modified.

### 3.2 Variance-Stabilization Rationale

Under global mean pooling $z_G = \frac{1}{N}\sum h_v$, the variance of the pooled representation contracts as $\mathcal{O}(1/N)$ if node embeddings are approximately independent. Normalizing by $\sqrt{N}$ instead scales the pooled representation such that if $\mathrm{Var}(h_v) \approx \sigma^2 I$, then $\mathrm{Var}(z_{\mathrm{norm}}) \approx \sigma^2 I$, independent of $N$. This is the Central-Limit-Theorem motivation for the $\sqrt{N}$ denominator.

### 3.3 Reference Distribution Fitting (id_train Only)

The reference mean $\mu_{\mathrm{norm}} \in \mathbb{R}^{128}$ and regularized covariance $\Sigma_{\mathrm{reg}} \in \mathbb{R}^{128 \times 128}$ were fitted **exclusively** on the $5{,}432$ intact `id_train` graphs using the project's established covariance procedure:

$$\Sigma = \frac{1}{N_{\mathrm{train}}-1}\sum_{i=1}^{N_{\mathrm{train}}}(z_{\mathrm{norm},i} - \mu)(z_{\mathrm{norm},i} - \mu)^T$$

$$\Sigma_{\mathrm{reg}} = (1 - \alpha)\,\Sigma + \alpha\,\frac{\mathrm{Tr}(\Sigma)}{d}\,I_{128}, \quad \alpha = 0.01$$

$$P_{\mathrm{norm}} = \mathrm{pinv}(\Sigma_{\mathrm{reg}})$$

No validation samples were used for fitting. The shrinkage rate $\alpha = 0.01$ is unchanged from the baseline and Variant A.

### 3.4 Covariance Numerical Health

| Property | Value | Assessment |
| :--- | :---: | :--- |
| Dimension | $128 \times 128$ | Exact match to $z_{\mathrm{norm}}$ |
| Symmetric ($\|P - P^T\|_F$) | $< 10^{-6}$ | Perfectly symmetric |
| Finite values | 100% finite (0 NaN, 0 Inf) | ✓ |
| Minimum eigenvalue ($\lambda_{\min}$) | **0.0035** | Strictly positive (PD) |
| Maximum eigenvalue ($\lambda_{\max}$) | **11.1692** | Bounded |
| Condition number ($\kappa$) | **3,187** | Well-conditioned |
| $\mathrm{Tr}(\Sigma_{\mathrm{reg}})/d$ | **7.4204** | vs. baseline 0.4488 — 16.5× larger (expected: $\sqrt{N}$-scaling inflates sum norms) |
| Regularization warnings | 0 | No jitter added |

> [!NOTE]
> The dramatically higher trace ($7.42$ vs. $0.45$ in the baseline) is expected and geometrically meaningful: $z_{\mathrm{norm}} = \frac{1}{\sqrt{N}}\sum h_v$ has norms proportional to $\sqrt{N} \cdot \|h_v\|$, so the coordinate variances are $\sim N$ times larger than under mean pooling. The shrinkage and pseudo-inverse procedures are numerically stable at this scale.

---

## 4. Variance Stabilization: Empirical Evidence

### Table 4.1: Total Embedding Variance by Graph-Size Bin (id_train Only)

| Size Bin (Atom Count) | $N_{\text{samples}}$ | Mean Nodes | **Mean-Pool** Total Var $\sum\mathrm{Var}(z_j)$ | **Norm-Pool** Total Var $\sum\mathrm{Var}(z_j)$ |
| :---: | :---: | :---: | :---: | :---: |
| $(3, 10]$ | 1,357 | 7.99 | **85.17** | 656.54 |
| $(10, 13]$ | 1,036 | 11.95 | 54.44 | 652.71 |
| $(13, 17]$ | 1,014 | 15.38 | 49.62 | 764.14 |
| $(17, 23]$ | 1,051 | 20.28 | 41.39 | 841.38 |
| $(23, 95]$ | 974 | 31.96 | **37.04** | **1,258.08** |

**Key Observation:** Mean pooling produces *decreasing* total variance as graph size increases ($85.17 \rightarrow 37.04$), confirming the $\mathcal{O}(1/N)$ variance compression. Normalized-sum pooling produces *increasing* total variance ($656.54 \rightarrow 1{,}258.08$), reflecting $\mathcal{O}(N)$ sum amplification — which **overshoots** the stabilization goal. Rather than achieving a constant variance across sizes, the $\frac{1}{\sqrt{N}}$ normalizer reverses the bias direction: large molecules are now dispersed far from $\mu_{\mathrm{norm}}$, while small molecules are compressed.

### Table 4.2: Overall Coordinate Variance Summary (id_train)

| Pooling Scheme | Mean Coord. Var | Std Coord. Var | Total Var $\sum\mathrm{Var}(z_j)$ |
| :--- | :---: | :---: | :---: |
| **Mean pooling (baseline)** | 0.449 | 0.352 | **57.44** |
| **Normalized-sum pooling** | 7.419 | 5.781 | **949.63** |

The normalized-sum representations have $\sim 16.5\times$ higher total variance than mean-pooled representations, concentrated in a size-dependent pattern that is amplified rather than removed.

### Table 4.3: Mean Embedding Norm by Size Bin (id_train)

| Size Bin | Mean Norm under Mean Pooling | Mean Norm under Norm-Sum Pooling |
| :---: | :---: | :---: |
| $(3, 10]$ | 14.33 | 39.91 |
| $(10, 13]$ | 13.08 | 45.19 |
| $(13, 17]$ | 12.97 | 50.84 |
| $(17, 23]$ | 12.59 | 56.64 |
| $(23, 95]$ | 12.37 | 69.21 |

Under mean pooling, the embedding norm is nearly constant across sizes (range $12.37–14.33$). Under normalized-sum pooling, the norm grows strongly with $N$ (range $39.91–69.21$), reflecting that $\|z_{\mathrm{norm}}\| = \frac{\|\sum h_v\|}{\sqrt{N}} \approx \sqrt{N} \cdot \|h_v\|$.

---

## 5. Benchmark Results

### Table 5.1: Complete Metric Comparison

| Benchmark | Cohort ($N_+$, $N_-$) | Metric | **Baseline** (Mean Pool, 128-dim) | **Variant B** (Norm-Sum, 128-dim) | **$\Delta$ (B − Base)** |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Scaffold Novelty (Raw)** | $N_+=408,\;N_-=678$ | **AUROC**<br>AUPR-Out<br>FPR@95 | **0.4491**<br>0.3321<br>0.9381 | **0.4354**<br>0.3358<br>0.9646 | **−0.0137**<br>+0.0037<br>+0.0265 |
| **Scaffold Novelty (Matched)** | $N_+=372,\;N_-=372$ | **AUROC**<br>AUPR-Out<br>FPR@95 | **0.4828**<br>0.4859<br>0.9516 | **0.4735**<br>0.4836<br>0.9704 | **−0.0093**<br>−0.0023<br>+0.0188 |
| **Dataset Novelty (Raw)** | $N_+=224,\;N_-=678$ | **AUROC**<br>AUPR-Out<br>FPR@95 | **0.4561**<br>0.2255<br>0.9248 | **0.5623**<br>0.2838<br>0.9130 | **+0.1063**<br>+0.0583<br>−0.0118 |
| **Dataset Novelty (Matched)** | $N_+=209,\;N_-=209$ | **AUROC**<br>AUPR-Out<br>FPR@95 | **0.5477**<br>0.5255<br>0.9139 | **0.5528**<br>0.5312<br>0.9234 | **+0.0052**<br>+0.0057<br>+0.0095 |

### Table 5.2: Score Distributions by Cohort

| Pooling | Cohort | Mean ± Std | Median [Q25, Q75] | Range [Min, Max] |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | `id_val` (678) | $110.34 \pm 141.47$ | $66.05\;[41.79, 112.60]$ | $[16.31,\;1765.79]$ |
| | `novel_scaffold_val` (408) | $86.00 \pm 112.71$ | $56.57\;[41.11,\;86.65]$ | $[19.45,\;1410.01]$ |
| | `novel_dataset_val` (224) | $94.16 \pm 155.26$ | $58.03\;[41.34,\;87.66]$ | $[21.89,\;1895.83]$ |
| **Variant B** | `id_val` (678) | $116.07 \pm 90.23$ | $89.55\;[66.57, 129.14]$ | $[20.04,\;878.05]$ |
| | `novel_scaffold_val` (408) | $102.45 \pm 93.32$ | $84.56\;[59.52, 106.83]$ | $[28.17,\;1131.73]$ |
| | `novel_dataset_val` (224) | $126.87 \pm 83.44$ | $96.60\;[74.15, 165.04]$ | $[25.12,\;595.65]$ |

**Distribution observations:**
- Under Variant B, `novel_dataset_val` has a clearly *higher* mean ($126.87$) and median ($96.60$) than `id_val` ($116.07$, $89.55$), reversing the baseline ordering ($94.16$ vs $110.34$). This drives the large AUROC gain on dataset novelty.
- Under Variant B, `novel_scaffold_val` still has a lower median ($84.56$) than `id_val` ($89.55$), yielding sub-0.50 raw scaffold AUROC.

---

## 6. Correlation Analysis

### Table 6.1: Pearson Correlation Between Atom Count $N$ and Novelty Score

| Cohort | $r(N, \text{Baseline})$ | $r(N, \text{Norm-Sum})$ | Sign Change? |
| :--- | :---: | :---: | :---: |
| `id_val` | **−0.3993** | **+0.2311** | ✓ Reversed |
| `novel_scaffold_val` | −0.2854 | +0.0538 | ✓ Near-zero |
| `novel_dataset_val` | −0.2919 | **+0.4226** | ✓ Reversed |
| All Val Combined | **−0.3389** | **+0.2312** | ✓ Reversed |

The $\frac{1}{\sqrt{N}}$ normalizer **reverses** the sign of the size-score correlation from strongly negative ($-0.3389$) to moderately positive ($+0.2312$). The correction overshoots: the normalizer does not achieve $r \approx 0$; it produces a new size bias in the opposite direction.

### Table 6.2: Pearson Correlation Between Baseline and Norm-Sum Scores (Same UIDs)

| Cohort | $r(\text{Base}, \text{Norm-Sum})$ |
| :--- | :---: |
| `id_val` | 0.5581 |
| `novel_scaffold_val` | 0.8446 |
| `novel_dataset_val` | 0.4558 |
| All Val Combined | 0.6094 |

The two scoring methods share moderate-to-high correlation, confirming they capture overlapping but distinct signals. The lower $r = 0.456$ on `novel_dataset_val` and $0.558$ on `id_val` explains why the relative ordering can shift substantially between methods.

---

## 7. Key Findings & Diagnostic Inferences

> [!IMPORTANT]
> These findings are exploratory, limited to a single seed, and should not be used to make causal or generalisation claims without further confirmation.

1. **$\sqrt{N}$ Normalization Does Not Stabilize Variance; It Amplifies in the Opposite Direction:**
   - Mean pooling produces total variance decreasing from $85.2$ to $37.0$ as $N$ increases (ratio $2.3\times$).
   - Normalized-sum pooling produces total variance increasing from $657$ to $1{,}258$ as $N$ increases (ratio $1.9\times$).
   - The $\sqrt{N}$ normalizer replaces a $1/N$ contraction with a $\sqrt{N}$ amplification. Neither achieves variance stability across graph sizes.

2. **Inconsistent Benchmark Behaviour — Opposite Direction for Scaffold vs. Dataset:**
   - Dataset novelty AUROC increases by $+0.1063$ (from $0.4561$ to $0.5623$), lifting above the 0.50 chance level.
   - Scaffold novelty AUROC *decreases* by $-0.0137$ (from $0.4491$ to $0.4354$), moving further below chance.
   - On matched controls (where size is already held constant), the change is negligible ($+0.0052$ on dataset, $-0.0093$ on scaffold), consistent with Variant A.

3. **Mechanistic Interpretation:**
   - Under normalized-sum, SIDER molecules (mean $30.2$ atoms, large) now have *higher* scores ($\bar{z} = 126.87$) than `id_val` molecules ($\bar{z} = 116.07$), because $z_{\mathrm{norm}} \propto \sqrt{N}$, and SIDER molecules are larger than `id_val` on average.
   - Tox21 scaffold-novel molecules (mean $22.1$ atoms) remain intermixed with `id_val` molecules (mean $22.5$ atoms) because their sizes are very similar; $\sqrt{N}$-scaling cannot separate their distributions.
   - This confirms that neither the $1/N$ contraction (baseline) nor the $\sqrt{N}$ amplification (Variant B) is correct: both create size-dependent biases because **molecular sizes differ structurally across the ID/OOD boundary**, not merely statistically.

4. **Score Direction Is Preserved:** Larger normalized-sum distance continues to mean more OOD. No post-hoc reversal was applied.

5. **What This Pilot Does and Does Not Establish:**
   - It confirms that size-dependent pooling artefacts are a genuine driver of the novelty ranking on the dataset benchmark.
   - It does not prove that correcting this artefact is achievable with a post-hoc pooling reparameterization applied to the frozen backbone.
   - It cannot be generalized to other seeds without replication; it is a single-seed exploratory pilot.

---

## 8. Limitations & Protocol Compliance

1. **Test Holdout Preserved:** No test graphs, labels, predictions, or checkpoints were accessed.
2. **Frozen Backbone:** Model weights were not retrained or fine-tuned. The normalized-sum transformation is applied post-forward-pass on the existing node embeddings.
3. **Single Seed (Seed 0):** No cross-seed generalization is claimed.
4. **No Threshold Tuning:** Score direction is fixed as higher = more OOD. No reversal was applied after seeing results.
5. **Baseline Unaltered:** The five-seed baseline benchmark summary and all prediction CSVs are unmodified.

---

## 9. Cautious Conclusion

The $\frac{1}{\sqrt{N}}$-normalized sum pooling transformation *partially* resolves the dataset novelty failure ($+0.1063$ AUROC, to $0.5623$) by reversing the sign of the size-score correlation. However, it *worsens* scaffold novelty ($-0.0137$, to $0.4354$) and does not achieve consistent improvement across both benchmarks. The empirical variance analysis shows that the normalizer overcorrects: instead of size-independent variance, it produces a $\sqrt{N}$-amplified variance pattern.

The result is consistent with the hypothesis that **the sub-0.50 novelty ranking is driven primarily by a systematic size difference between `id_val` and `novel_dataset_val`** (mean atoms: 22.5 vs. 30.2), which is a property of this specific benchmark's sample composition rather than a model failure mode that can be corrected by reparameterizing the pooling aggregator.

For scaffold novelty, where the Tox21-novel molecules and `id_val` molecules are nearly identical in size (22.1 vs. 22.5 atoms), no post-hoc pooling reparameterization is expected to rescue AUROC because the failure stems from the scaffold-disjoint split: `id_val` itself has unseen scaffolds relative to `id_train`.

---

## 10. Recommendation

The most informative next step to distinguish the two confounders (size bias vs. scaffold-disjoint split) is:

### **Controlled ID-Reference Replacement Pilot**
- Keep the backbone and all model weights frozen at Seed 0.
- Instead of scoring `novel_scaffold_val` against `id_val` (which has unseen scaffolds), identify a subset of `id_val` whose scaffolds overlap with `id_train` (approximately 26.6% of `id_val` per the scaffold audit) and score `novel_scaffold_val` exclusively against this scaffold-matched ID reference.
- If scaffold-AUROC improves substantially using only the scaffold-overlapping ID reference, the scaffold-disjoint split is confirmed as the dominant cause.
- If scaffold-AUROC remains sub-0.50 even on the scaffold-matched ID reference, the failure is attributable to the learned representation itself, independent of split construction.
