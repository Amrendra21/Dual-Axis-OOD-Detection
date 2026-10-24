# Dual-Axis Out-of-Distribution Detection on Molecular Graphs: A Diagnostic Benchmarking Study of Physical Validity and Scaffold Novelty

**Authors:** Member 5 (Evaluation, Benchmarking & Manuscript Lead) on behalf of the Dual-Axis OOD Detection Team  
**Date:** October 2026  
**Status:** Complete Faculty-Review-Ready Research Paper (Evidence-Frozen / Pre-Submission Final)  
**Approved Research Scope:** Option A — Diagnostic & Benchmarking Study (No Claims of Algorithmic Superiority)  

---

## Abstract

Graph Neural Networks (GNNs) deployed for molecular property prediction and virtual screening fail unpredictably when evaluated on out-of-distribution (OOD) molecular graphs. Conventional OOD detection benchmarks collapse all distributional shifts into a single scalar anomaly score, conflating two fundamentally distinct failure modes: **Physical / Structural Chemical Validity** (violations of covalent valence constraints, such as pentavalent "Texas carbons") and **Semantic Scaffold Novelty** (synthesizable, chemically valid molecules featuring core ring scaffolds unseen during training).

In this work, we formalize and benchmark a **Dual-Axis OOD Detection framework** for molecular graphs using the curated `processed_v3` benchmark (10,140 graphs spanning the Tox21 toxicology challenge and SIDER clinical database). Axis 1 evaluates structural chemical validity across controlled graph corruption families, while Axis 2 evaluates semantic novelty under a hybrid split (ring-containing molecules split by Bemis-Murcko scaffold; ring-free molecules split randomly) alongside cross-domain dataset shift (SIDER). We present a comprehensive multi-seed validation benchmark comparing a unified multi-task architecture (DAG-NN, Track A: strict zero corruption exposure), decoupled validity and novelty models (Member 4, Track B: outlier exposure on corrupted graphs), multi-seed single-score baselines (S1 Energy, S2 Mahalanobis), and non-learning size controls.

Our empirical findings establish three core insights:
1. **Physical Valence Detection:** GNN message passing reliably detects physical valence violations without explicit chemical rule engines, achieving validation AUROC $0.9584 \pm 0.0119$ under Track A, which receives no corrupted graphs during training, and $0.9966$ under outlier exposure Track B, substantially outperforming the atom-count control on Family A (AUROC $0.5107$).
2. **Failure-Mode Conflation:** Single-score OOD detectors conflate validity violations with scaffold novelty near chance level on the evaluated validation cohorts (Task X cause-separation AUROC $0.5655 \pm 0.0104$ for Energy, $0.5677 \pm 0.0101$ for Mahalanobis), whereas dual-axis scoring cleanly separates the two failure modes (Task X AUROC $0.9073 \pm 0.0069$). Critically, however, joint 3-class triage (Task D) reveals that the resulting macro recall ($0.5377 \pm 0.0182$) masks a severe per-class imbalance: invalid detection is robust (recall $67.62\% \pm 5.66\%$) while novel-chemotype detection is near-zero (recall $2.09\% \pm 0.52\%$), because inverted novelty scores displace the operational threshold above the median novel molecule score.
3. **Confounders in Molecular OOD Benchmarks:** Standard distance-based novelty estimators yield sub-0.50 raw AUROC on held-out scaffold validation ($0.4539 \pm 0.0061$ across 5 seeds). The diagnostic results indicate that reference-distribution shift contributes to the observed novelty-ranking failure: in-distribution validation molecules are themselves scaffold-disjoint from training (0.0% Murcko scaffold overlap), dispersing them into the representation periphery (mean distance 7.63 vs. 6.75 for novel molecules). Re-centering the density estimator on an in-distribution reference sharing generic Murcko framework topology elevates AUROC to $0.7440$, while separation against an acyclic reference ($1.0000$) exposes a dominant cyclic-vs-acyclic topological confound. Furthermore, apparent raw gains on cross-domain novelty (SIDER) are heavily confounded by heavy atom count disparities (atom count alone reaches AUROC $0.6280$), collapsing to chance level ($0.5024$) under size- and ring-matched controls.

This diagnostic study demonstrates the necessity of decoupled failure-mode evaluation, rigorous matched controls, and careful reference alignment in molecular graph OOD benchmarking.

---

## 1. Introduction

Deep learning on molecular graphs has become a cornerstone of computational chemistry, enabling property prediction, bioactivity profiling, and virtual screening directly from molecular topologies $G = (V, E)$. Despite impressive benchmark accuracy on in-distribution test sets, graph neural networks (GNNs) are notoriously brittle under distributional shift: when queried on inputs outside their training support, standard architectures frequently emit overconfident, erroneous predictions.

In computer vision and natural language processing, out-of-distribution (OOD) detection typically addresses semantic class novelty or domain style shift. In molecular chemistry, however, distributional shift exhibits a unique dual nature:
1. **Axis 1 — Structural / Syntactic Validity:** Generative algorithms, de novo molecular design pipelines, and graph perturbation techniques can produce graphs that violate fundamental physical valence constraints (e.g., carbon possessing 5 covalent bonds, colloquially termed a "Texas carbon"). A robust molecular model must recognize non-physical graphs as chemically invalid.
2. **Axis 2 — Semantic / Scaffold Novelty:** Medicinal chemistry frequently explores uncharted chemical space, where synthesized molecules are physically valid and synthesizable, but possess core ring scaffolds never encountered during training.

Treating these distinct axes as a single scalar OOD problem obscures whether a detector fails due to physical topological blind spots or lack of chemical semantics. A single high anomaly score cannot inform a medicinal chemist whether a candidate molecule represents a viable novel therapeutic chemotype or an unphysical graph artifact.

To address this challenge, we present a **Diagnostic Benchmarking Study (Option A)** on the curated `processed_v3` molecular graph dataset. We do not claim to introduce a universally superior OOD detector. Instead, we establish a standardized, zero-leakage evaluation protocol to investigate two central scientific questions:
* **Question 1:** Can unified graph architectures distinguish chemically impossible graphs from novel chemotypes without explicit rule engines, and how do training regimes (unsupervised Track A vs. outlier exposure Track B) impact detection?
* **Question 2:** Do conventional graph OOD benchmarks suffer from hidden topological and size confounders that distort measured performance?

---

## 2. Related Work

### 2.1 Graph Neural Networks for Molecular Representation
Message passing neural networks (MPNNs) iteratively aggregate neighborhood embeddings:
$$h_v^{(k)} = \text{UPDATE}^{(k)}\left(h_v^{(k-1)}, \text{AGGREGATE}^{(k)}\left(\{h_u^{(k-1)} : u \in \mathcal{N}(v)\}\right)\right)$$
Key architectures include Graph Convolutional Networks (GCN) [8], Graph Isomorphism Networks (GIN) [13], and edge-aware variants such as GINE [5]. While standard message passing is permutation-invariant, it does not inherently encode physical chemistry valence conservation.

### 2.2 Out-of-Distribution Detection in Machine Learning
Baseline post-hoc OOD detection relies on Maximum Softmax Probability (MSP) [3]. Subsequent developments utilize energy-based scores [11], feature-space Mahalanobis distances [10], and outlier exposure (OE) [4]. In molecular graphs, recent benchmarks such as GOOD [2] and DrugOOD [7] examine covariate and scaffold shifts, but they collapse physical validity and semantic novelty into a single evaluation axis.

### 2.3 Molecular Splitting and Chemical Validity
Random splitting in molecular benchmarks causes severe analog leakage. Bemis-Murcko scaffold clustering [1] separates distinct chemical series. However, when evaluation sets are made scaffold-disjoint from training, the reference in-distribution validation set is itself composed of novel scaffolds, creating subtle distribution shifts for distance-based density estimators that have remained largely uncharacterized.

---

## 3. Problem Formulation, Tasks & Training Regimes

Let a molecular graph be $G = (V, E)$, where $v \in V$ represents an atom with feature vector $x_v \in \mathbb{R}^{30}$, and $e_{uv} \in E$ represents a chemical bond with feature vector $e_{uv} \in \mathbb{R}^6$. Let $\mathcal{M}$ denote the set of all physically valid chemical molecules conforming to covalent valence rules.

```
                          AXIS 1: Structural Validity
                          (Physical Chemical Rules)
                                   ▲
                                   │
              Chemically Invalid   │   Compound Anomalies
              (Corrupted Graphs)   │   (Untested Boundary)
                                   │
      ─────────────────────────────┼─────────────────────────────►
          In-Distribution (ID)     │   AXIS 2: Semantic Novelty
          (Known Scaffolds)        │   (New Scaffolds / SIDER)
                                   │
                                   ▼
```

### 3.1 Taxonomy of Graph Partitions
* **Definition 1 (In-Distribution, $\mathcal{D}_{\text{in}}$):** Chemically valid graphs ($G \in \mathcal{M}$) drawn from the Tox21 assay training distribution whose Bemis-Murcko scaffold belongs to the training scaffold set $\mathcal{S}_{\text{train}}$.
* **Definition 2 (Axis 1: Structural / Syntactic OOD, $\mathcal{D}_{\text{syn}}$):** Graphs $G \notin \mathcal{M}$ violating physical covalent valence constraints (specifically carbon valence $>4$ or oxygen valence $>2$).
* **Definition 3 (Axis 2: Semantic / Scaffold OOD, $\mathcal{D}_{\text{sem}}$):** Chemically valid molecules ($G \in \mathcal{M}$) whose core scaffold belongs to a disjoint held-out set $\mathcal{S}_{\text{ood}} \cap \mathcal{S}_{\text{train}} = \emptyset$.
* **Definition 4 (Axis 2: Cross-Domain Dataset Shift, $\mathcal{D}_{\text{far}}$):** Chemically valid approved pharmaceutical molecules drawn from SIDER.
* **Scope Boundary (Compound Anomalies):** Graphs that are simultaneously invalid and novel ($\mathcal{D}_{\text{syn}} \cap \mathcal{D}_{\text{sem}}$) do not exist in `processed_v3`. Task D evaluates triage between distinct failure modes, not compound anomalies.

### 3.2 Training Regimes (Track Disclosures)
* **Track A (Strict Zero-Exposure):** Models are trained strictly on `id_train` with Tox21 multi-task assay labels and masked feature reconstruction. Models receive zero exposure to corrupted graphs during training.
* **Track B (Outlier Exposure):** Models are trained on `id_train` plus `corrupt_train` (Families A & B labeled as invalid). Track B models must not be characterized as unsupervised.

### 3.3 Evaluation Tasks Formalized
1. **Task V1 (Validity vs. ID):** Evaluates `validity_score` on invalid graphs vs. `id_val` ($0 = \text{ID}, 1 = \text{Invalid}$). Reported per family and ALL.
2. **Task V2 (Validity vs. All-Valid):** Evaluates `validity_score` on invalid graphs vs. `id_val` + novel graphs ($0 = \text{Valid}, 1 = \text{Invalid}$). Ensures novel valid molecules are not falsely flagged as invalid.
3. **Task N (Raw Novelty):** Evaluates `novelty_score` on novel graphs vs. `id_val` ($0 = \text{ID}, 1 = \text{Novel}$).
4. **Task NM (Matched Controls):** Evaluates `novelty_score` on novel graphs vs. size- and ring-matched ID validation controls ($0 = \text{Matched ID}, 1 = \text{Novel}$).
5. **Task S (Single-Score Detection):** Evaluates a single scalar score on all OOD inputs vs. `id_val`.
6. **Task X (Cause Separation):** Evaluates `validity_score` or `single_score` on corrupted graphs vs. novel graphs ($0 = \text{Novel}, 1 = \text{Invalid}$).
7. **Task D (Joint 3-Class Triage):** Applies 95th-percentile thresholds $(\tau_v, \tau_n)$ derived from `id_val` to assign instances: Invalid if $S_v \ge \tau_v$; else Novel if $S_n \ge \tau_n$; else ID.

---

## 4. Dataset Construction, Controlled Corruption & Integrity

We utilize the authoritative `processed_v3` dataset package, comprising **10,140 total molecular graphs** curated from the Tox21 toxicology challenge [6] and SIDER clinical database [9], both widely adopted from the MoleculeNet molecular machine learning benchmark collection [12].

### 4.1 Curation & Hybrid Splitting Scheme
From raw SMILES repositories, inorganic salts, unparseable structures, and duplicate SMILES were removed. Rather than a purely scaffold-disjoint split across all molecules, `processed_v3` enforces a **hybrid splitting protocol**:
* **Ring-Containing Molecules (75% of ID):** Partitioned strictly by Bemis-Murcko scaffold clusters to prevent chemotype leakage.
* **Ring-Free / Acyclic Molecules (25% of ID):** Assigned randomly across training, validation, and test splits.

Authoritative validation counts after applying the exclusion policy (`exclude == 0`) from `tables/exclusions.csv` are detailed in Table 1. Exactly two validation molecules were excluded: `SID-00482` (exact graph-level duplicate of a training molecule, identified by identical model input tensor) and `T21-02289` (exact graph-level duplicate across sets, sharing identical model input with multiple training graphs that contain non-standard element features).

### Table 1: Authoritative Dataset Structure and Validation Populations (`processed_v3`)

| Split / Set Name | Raw Graphs | Excluded Graphs | Active Evaluation Graphs | Underlying Class / Axis | Role in Benchmark |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `id_train` | 5,432 | 0 | 5,432 | In-Distribution (Tox21) | Training support (all tracks) |
| `id_val` | 679 | 1 (`T21-02289`) | **678** | In-Distribution Validation | Validation negative reference |
| `corrupt_val` (Family A) | 272 | 0 | **272** | Axis 1: Bond Order Corruptions | Validation positive (Texas carbon) |
| `corrupt_val` (Family B) | 271 | 0 | **271** | Axis 1: Bond Relocation Corruptions | Validation positive (Relocation) |
| `novel_scaffold_val` | 408 | 0 | **408** | Axis 2: Held-out Scaffold Families | Validation positive (Tox21 novel) |
| `novel_dataset_val` | 225 | 1 (`SID-00482`) | **224** | Axis 2: Cross-Domain Novelty (SIDER) | Validation positive (SIDER novel) |
| Matched Scaffold Pairs | 372 pairs | 0 | **372 pairs** | Task NM Matched Controls | Size/ring-matched validation |
| Matched Dataset Pairs | 209 pairs | 0 | **209 pairs** | Task NM Matched Controls | Size/ring-matched validation |

### 4.2 Controlled Graph Corruption Families
* **Family A (Bond Order Increment):** Inserts an additional covalent bond between adjacent carbon atoms, driving carbon valence to 5 ("Texas carbon") while freezing hydrogen counts. Atom composition and degree sequence are perfectly preserved.
* **Family B (Bond Relocation):** Relocates an existing covalent bond onto an already saturated carbon atom. Alters local degree distribution.
* **Family C (Heteroatom Substitution — Held-Out Test Only):** Replaces a neutral divalent/tetravalent atom with an oxygen while preserving bonds, violating oxygen valency. Quarantined strictly in `corrupt_test` ($N=181$); completely absent from validation.

### 4.3 Matched-Control Caliper (Task NM)
To eliminate molecular size and ring presence as trivial detection shortcuts, `matched_controls_filtered.csv` pairs each novelty graph with an ID validation graph satisfying:
1. Identical ring status (`is_acyclic` matches exactly).
2. Heavy atom count difference $|\Delta\text{atoms}| \le 2$.
3. Bond count difference $|\Delta\text{bonds}| \le 2$.

Greedy matching covers **91.2%** of validation scaffold novelty graphs (372 pairs) and **93.3%** of validation dataset novelty graphs (209 pairs).

---

## 5. Standardized Evaluation Protocol & Metrics

Given ground-truth binary label $y \in \{0, 1\}$ ($0 = \text{ID/Negative}$, $1 = \text{OOD/Positive}$) and predicted anomaly score $S(G) \in \mathbb{R}$ where higher values indicate greater anomaly:
1. **AUROC:** Area under the Receiver Operating Characteristic curve, measuring rank separation across all thresholds.
2. **AUPR-Out:** Area under the Precision-Recall curve with OOD treated as positive ($y = 1$).
3. **AUPR-In:** Area under the Precision-Recall curve with ID treated as positive ($y = 0$, evaluated via $-S$).
4. **FPR@95%TPR:** False Positive Rate on ID negatives at the operational decision threshold $\tau$ achieving $\ge 95\%$ True Positive Rate on OOD positives.

All hyperparameters, early stopping checkpoints, and decision thresholds were derived strictly on validation partitions. The test partition was kept strictly quarantined.

---

## 6. Experimental Results & Benchmarking

All reported numerical results are reproduced directly from verified validation artifacts.

### Table 2: Primary Validation Benchmark Performance Across Tasks (Validation Partitions Only)

| Model Architecture | Training Track | Task V1 AUROC (Fam A) | Task V1 AUROC (Fam B) | Task V2 AUROC (All Valid) | Task N AUROC (Scaffold Raw) | Task NM AUROC (Scaffold Matched) | Task N AUROC (Dataset Raw) | Task NM AUROC (Dataset Matched) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DAG-NN** (5 seeds) | Track A | **0.9584 $\pm$ 0.0119** | **0.8565 $\pm$ 0.0083** | **0.9074 $\pm$ 0.0071** | **0.4539 $\pm$ 0.0061** | **0.4844 $\pm$ 0.0046** | **0.4521 $\pm$ 0.0047** | **0.5488 $\pm$ 0.0047** |
| **Member 4 GINE/Mahal** | Track B | **0.9966** | **0.9952** | [Unavailable]* | **0.4570** | **0.4783** | **0.6104** | **0.5378** |
| **C2natoms Control** | N/A | 0.5107 | 0.5281 | 0.4969 | 0.5033 | 0.5034 | **0.6280** | 0.5024 |
| **C1 Histogram GBDT** | N/A | **0.8890** | 0.6570 | [Unavailable] | **0.7520** | **0.6890** | **0.7280** | **0.6640** |

*\*Note: Member 4 handoff lacked cross-axis validity predictions on novel graphs, precluding Task V2 evaluation. Uncertainty is reported as $\pm$ standard deviation across 5 seeds for DAG-NN; Member 4 and non-learning baselines are single-run evaluations.*

### Table 3: Single-Score Conflation vs. Dual-Axis Cause Separation (Task X) Across 5 Seeds

| Evaluation Method | Training Track | Task S AUROC (All OOD vs. ID) | Task X AUROC (Invalid Fam A vs. Novel) | Task X AUROC (Invalid ALL vs. Novel) | Task D Macro Recall | Task D Novel Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1 Energy GIN** | Track A | 0.4790 $\pm$ 0.0027 | 0.5344 $\pm$ 0.0133 | 0.5655 $\pm$ 0.0104 | N/A (Single score) | N/A |
| **S2 Mahalanobis GIN** | Track A | 0.4988 $\pm$ 0.0069 | 0.5799 $\pm$ 0.0077 | 0.5677 $\pm$ 0.0101 | N/A (Single score) | N/A |
| **DAG-NN (Dual-Axis)** | Track A | N/A (Dual outputs) | **0.9587 $\pm$ 0.0097** | **0.9073 $\pm$ 0.0069** | **0.5377 $\pm$ 0.0182** | **0.0209 $\pm$ 0.0052** |

### 6.1 Analysis of Physical Valence Detection (Axis 1)
Under Track A (zero exposure to corruptions), DAG-NN achieves AUROC **$0.9584 \pm 0.0119$** on Family A and **$0.8565 \pm 0.0083$** on Family B. Under Track B (outlier exposure), Member 4's GINE validity detector reaches AUROC $> 0.995$. Both learning models dramatically outperform graph-size controls (C2natoms AUROC $0.5107$). However, the C1 histogram probe demonstrates that raising bond orders produces global bond-type imbalances that a boosted tree can partially detect (AUROC $0.8890$) without local message passing.

### 6.2 Analysis of Failure-Mode Separation (Task X)
Single-score OOD baselines (S1 Energy and S2 Mahalanobis) fail to separate invalid graphs from novel graphs on the evaluated cohorts, yielding Task X AUROCs of **$0.5655$** and **$0.5677$** (near chance). Because a single scalar score treats any deviation from training as anomalous, single-score detectors cannot distinguish physical corruption from chemical novelty. In contrast, DAG-NN's dual-axis separation achieves Task X AUROC **$0.9073 \pm 0.0069$**.

### 6.3 Analysis of 3-Class Triage (Task D)
Under the operational 3-class decision rule, DAG-NN achieves a macro recall of **$0.5377 \pm 0.0182$**. While invalid recall is robust (**$67.62\% \pm 5.66\%$**), novel recall is severely depressed (**$2.09\% \pm 0.52\%$**). Because raw scaffold novelty scores are inverted, the 95th-percentile threshold $\tau_n$ derived from `id_val` sits above the median of the novel molecules, resulting in high false negative rates for novel chemotypes.

---

## 7. Discussion & Forensic Diagnostic Analysis

### 7.1 Mechanism of Scaffold Novelty Failure ($\text{AUROC} < 0.50$)
Both DAG-NN (AUROC $0.4539$) and Member 4 (AUROC $0.4570$) perform below random guessing on raw scaffold novelty. Forensic investigation confirms that the sub-0.50 AUROC does not arise from an inverted scoring convention. Instead, the diagnostic results indicate that reference-distribution shift contributes to the observed novelty-ranking failure:
* **Scaffold-Disjoint Split Drift:** In `processed_v3`, `id_val` contains 509 unique cyclic scaffolds, **0.0% of which appear in `id_train`**.
* **Embedding Geometry:** Because the Gaussian centroid $\mu_{\text{train}}$ was fitted strictly on `id_train`, `id_val` molecules disperse far into the periphery (mean Euclidean distance to $\mu_{\text{train}} = 7.63$, total variance $63.35$), exceeding the dispersion of curated `novel_scaffold_val` molecules (mean distance $6.75$, total variance $42.61$). Evaluating novel molecules against `id_val` compares two distinct sets of out-of-training scaffolds, contributing to the inverted ranking.

### 7.2 Molecular Size Confounding & Pooling Dynamics
* **Variant A Pilot ($[z_G \parallel \log(N)]$):** Appending a linear size coordinate left scaffold AUROC at $0.4454$ ($\Delta = -0.0037$), because the $1/N$ variance contraction of mean pooling operates across all 128 hidden dimensions.
* **Variant B Pilot ($\frac{1}{\sqrt{N}}\sum h_v$):** Normalized-sum pooling reversed size correlation from negative to positive ($r = +0.231$), lifting dataset AUROC to $0.5623$ but worsening scaffold AUROC to $0.4354$.
* **Size Bias in Cross-Domain Novelty:** On raw SIDER, C2 atom count achieves AUROC **$0.6280$**, collapsing to **$0.5024$** under matched controls. Member 4 collapses from **$0.6104$** to **$0.5378$**, confirming that raw cross-domain gains are heavily confounded by molecular size disparities.

### 7.3 Controlled Reference Replacement Diagnostic
To diagnostically evaluate whether the frozen GNN representation $z_G$ possesses structural novelty discrimination capacity, a controlled reference replacement pilot (single seed, exploratory) evaluated three ID reference conditions:

### Table 4: Controlled ID-Reference Replacement Diagnostic Pilot (Seed 0 Frozen Weights)

| Reference Condition | Negative Cohort ($N$) | Target Novelty Set ($N$) | Paradigm A AUROC (Frozen $\text{id}_{\text{train}}$) | Paradigm B AUROC (Re-estimated) | Paradigm B AUPR-Out | Diagnostic Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Active ID (`id_val`)** | 678 | 408 | 0.4491 | 0.5219 | 0.3894 | Eliminates train-to-val shift; centers ID distribution |
| **Generic Murcko Overlap** | 190 | 408 | 0.4395 | **0.7440** | **0.8697** | **Exploratory indication under matched framework** |
| **Acyclic / Ring-Free Reference** | 169 | 408 | 0.2056 | **1.0000** | **1.0000** | **Topological cycle confounding (trees vs. rings)** |

* **Paradigm B Generic Overlap (AUROC $0.7440$):** When the density estimator is re-centered on the 190 ID validation molecules that share generic carbon frameworks with training, novel molecules are recognized as geometric outliers (mean distance $164.9$ vs. $74.0$). This exploratory, single-seed result suggests that the frozen representation may contain structural information useful for distinguishing the selected novel molecules under this particular reference condition. Because the reference cohort is restricted and the density estimator is re-estimated, this result does not establish general scaffold-novelty detection.
* **Paradigm B Acyclic Reference (AUROC $1.0000$):** Centering on ring-free ID molecules achieves perfect separation (score gap $+169.5$). This is explicitly disclosed as **topological cycle detection rather than scaffold novelty detection**: 100% of novel molecules contain rings, while 100% of acyclic reference molecules are trees.

---

## 8. Limitations, Threats to Validity & Reproducibility Boundaries

1. **Benchmark Reference Shift:** Scaffold-disjoint splitting in validation inherently penalizes distance-based density estimators fitted on training data.
2. **Compound Anomaly Scope:** `processed_v3` contains no simultaneous corrupted-novel molecules; compound failure modes remain uncharacterized.
3. **Member 4 Checkpoint Boundaries:** Model weights for Member 4 were unbundled in the repository handoff. Member 4 results are prediction-level verified only. Cross-axis prediction scores were absent, preventing Task D evaluation on Member 4.
4. **Validation-Only Evidence Scope:** All reported results, threshold selections, and diagnostic pilots are strictly validation-only. The final test partition remains quarantined.

---

## 9. Conclusion

This diagnostic benchmarking study formalizes the Dual-Axis OOD Detection paradigm for molecular graphs. By decoupling physical chemical validity from semantic novelty, we demonstrate that GNNs can reliably identify physical valence corruptions without explicit rule engines, resolving the failure-mode conflation of single-score detectors (improving cause separation from $0.56$ to $>0.90$). However, joint 3-class triage (Task D) exposes a critical limitation: the macro recall of $0.5377 \pm 0.0182$ conceals near-zero novel-chemotype detection (recall $2.09\% \pm 0.52\%$), because the inverted novelty score distribution displaces the operational threshold beyond the reach of most novel molecules. Furthermore, our forensic diagnostic pilots indicate that reference-distribution shift from scaffold-disjoint validation partitioning, molecular size disparities, and topological cycle confounding contribute substantially to the observed novelty-ranking behaviors. Establishing reliable molecular AI requires moving beyond single scalar OOD metrics toward multi-axis evaluation with rigorous matched controls, operational thresholds calibrated for each axis independently, and honest per-class reporting.

---

## References

1. Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. *Journal of Medicinal Chemistry*, 39(15), 2887-2893.
2. Gui, S., Li, X., Wang, L., & Ji, S. (2022). GOOD: A graph out-of-distribution benchmark. *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 2059-2073.
3. Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *International Conference on Learning Representations (ICLR)*.
4. Hendrycks, D., Mazeika, M., & Dietterich, T. (2019). Deep anomaly detection with outlier exposure. *International Conference on Learning Representations (ICLR)*.
5. Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., & Leskovec, J. (2020). Strategies for pre-training graph neural networks. *International Conference on Learning Representations (ICLR)*.
6. Huang, W. et al. (2021). Tox21 Data Challenge: High-throughput screening for toxicology. *Frontiers in Environmental Science*, 9, 602606.
7. Ji, Y., Zhang, L., Wang, J., Wu, B., Huang, L., Xu, T., Rong, Y., Li, F., & Tsung, F. (2023). DrugOOD: Out-of-distribution (OOD) dataset curator and benchmark for AI-aided drug discovery. *International Conference on Learning Representations (ICLR)*.
8. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR)*.
9. Kuhn, M., Letunic, I., Jensen, L. J., & Bork, P. (2016). The SIDER database of drugs and side effects. *Nucleic Acids Research*, 44(D1), D1075-D1079.
10. Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *Advances in Neural Information Processing Systems (NeurIPS)*, 31.
11. Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based out-of-distribution detection. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 21464-21475.
12. Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Hao, C., Leswing, K., Xie, Z., Yao, V., & Pande, V. S. (2018). MoleculeNet: A benchmark for molecular machine learning. *Chemical Science*, 9(2), 513-530.
13. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *International Conference on Learning Representations (ICLR)*.
