# Dual-Axis Out-of-Distribution Detection on Molecular Graphs: Disentangling Structural Validity from Semantic Novelty

**Authors:** TODO: [Author Names, Affiliations, and Contact Information]
**Date:** October 2026
**Status:** Working Draft / Research Paper Skeleton

---

## Abstract

Graph Neural Networks (GNNs) have achieved widespread adoption in molecular property prediction and virtual screening. However, their deployment in safety-critical drug discovery pipelines is hampered by unpredictable failures on out-of-distribution (OOD) inputs. Conventional OOD detection benchmarks collapse all distributional shifts into a single scalar anomaly score, conflating two fundamentally distinct failure modes: **Structural / Syntactic Validity** (physical chemistry violations, such as hypervalent "Texas carbons") and **Semantic Novelty** (chemically valid molecules featuring entirely unseen core scaffolds).

In this work, we formalize a **Dual-Axis OOD Detection Benchmark** for molecular graphs based on the Tox21 bioactivity dataset (7,823 curated molecules). Axis 1 evaluates structural validity through controlled graph corruption that introduces pentavalent carbons violating covalent valency constraints, while Axis 2 evaluates semantic novelty via strict Bemis-Murcko scaffold partitioning (70/10/10/10 split). We establish a rigorous evaluation protocol reporting AUROC, AUPR-Out, AUPR-In, and FPR@95%TPR, along with combined dual-axis metrics. Furthermore, we define a standardized experimental evaluation protocol designed for representative GNN architectures (GCN, GIN, GAT) and baseline models across post-hoc scoring functions (MSP, Energy-based, Mahalanobis distance), currently registered as planned benchmark experiments. This framework provides a principled foundation to measure whether models can reliably differentiate both chemically impossible graphs and novel therapeutic chemotypes from known in-distribution molecules.

---

## 1. Introduction

Deep learning on graphs has revolutionized computational chemistry, enabling molecular representation learning directly from molecular graph topologies $G = (V, E)$. Despite impressive benchmark accuracies, deep learning models are notoriously vulnerable to distributional shift: when queried on inputs outside their training support, standard architectures tend to yield overconfident, erroneous predictions.

In computer vision and NLP, out-of-distribution (OOD) detection typically addresses covariate shift or novel semantic classes. In molecular chemistry, however, distributional shift exhibits a unique dual nature:
1. **Axis 1 — Structural / Syntactic Validity:** Generative algorithms and graph perturbation techniques can produce graphs that violate fundamental physical laws (e.g., carbon possessing 5 covalent bonds, colloquially termed a "Texas carbon"). A molecular model must recognize non-physical graphs as chemically invalid.
2. **Axis 2 — Semantic / Scaffold Novelty:** Medicinal chemistry frequently explores uncharted chemical space, where synthesized molecules are physically valid but possess ring systems or scaffolds never encountered in the training distribution.

Treating these distinct axes as a single OOD problem obscures whether a detector fails due to topological blind spots or lack of chemical semantics. To address this challenge, we introduce the **Dual-Axis OOD Detection Benchmark**:
- We curate the **Tox21** benchmark into an audited, zero-leakage split separating known scaffolds, disjoint semantic OOD scaffolds, and valency-corrupted graphs.
- We show via non-parametric hypothesis testing (Mann-Whitney U tests, Cohen's $d < 0.11$) that neither axis can be solved by trivial size or property shortcuts.
- We formalize a standardized evaluation framework with threshold-free and safety-critical metrics.

---

## 2. Related Work

### 2.1 Graph Neural Networks for Molecular Representation
Message passing neural networks (MPNNs) iteratively aggregate neighborhood embeddings:
$$h_v^{(k)} = \text{UPDATE}^{(k)}\left(h_v^{(k-1)}, \text{AGGREGATE}^{(k)}\left(\{h_u^{(k-1)} : u \in \mathcal{N}(v)\}\right)\right)$$
Key architectures include Graph Convolutional Networks (GCN) [Kipf & Welling, 2017], Graph Isomorphism Networks (GIN) [Xu et al., 2019], and Graph Attention Networks (GAT) [Veličković et al., 2018]. While expressive, standard message passing does not inherently enforce physical valence rules.

### 2.2 Out-of-Distribution Detection in Machine Learning
Baseline post-hoc OOD detection relies on Maximum Softmax Probability (MSP) [Hendrycks & Gimpel, 2017]. More recent advances employ energy-based scores [Liu et al., 2020], feature-space Mahalanobis distances [Lee et al., 2018], and virtual outlier synthesis (VOS) [Du et al., 2022]. However, these methods have primarily been evaluated on image domains (e.g., CIFAR-10 vs. SVHN) and remain underexplored on dual-axis graph shifts.

### 2.3 Molecular Splitting and Chemical Validity
Random dataset splitting suffers from analog bias, where identical chemotypes appear in both training and test sets. Bemis-Murcko scaffold clustering [Bemis & Murcko, 1996] enforces scaffold separation. However, existing benchmarks rarely evaluate physical validity violations alongside scaffold shifts.

---

## 3. Problem Formulation & Dual-Axis Taxonomy

Let a molecular graph be $G = (V, E)$, where $v \in V$ represents an atom with feature vector $x_v$, and $e_{uv} \in E$ represents a chemical bond. Let $\mathcal{M}$ denote the set of all valid chemical molecules conforming to valence constraints.

```
                          AXIS 1: Structural Validity
                          (Physical Chemical Rules)
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

### Definition 1 (In-Distribution, $\mathcal{D}_{\text{in}}$)
Molecules $(G, y) \sim P_{\text{in}}$ that are chemically valid ($G \in \mathcal{M}$) and whose Bemis-Murcko scaffold $S(G)$ belongs to the training scaffold set $\mathcal{S}_{\text{train}}$.

### Definition 2 (Axis 1: Structural / Syntactic OOD, $\mathcal{D}_{\text{syn}}$)
Graphs $G \notin \mathcal{M}$ that violate physical chemistry valence rules (specifically, carbon valence $> 4$). These graphs fail standard chemical sanitization (e.g., RDKit `Chem.SanitizeMol`).

### Definition 3 (Axis 2: Semantic / Scaffold OOD, $\mathcal{D}_{\text{sem}}$)
Molecules $G \in \mathcal{M}$ that are chemically valid, but whose scaffold belongs to a disjoint set $\mathcal{S}_{\text{ood}} \cap \mathcal{S}_{\text{train}} = \emptyset$.

### Definition 4 (Planned Far-OOD Semantic Benchmark: ID vs. SIDER, $\mathcal{D}_{\text{far}}$)
Molecules $G \in \mathcal{M}$ drawn from an external chemical and clinical repository (such as approved drugs with adverse reaction profiles in SIDER) evaluated against $\mathcal{D}_{\text{in}}$ to measure model degradation under cross-domain semantic shift. Planned evaluation task; exact split counts and curation details: TODO: [Pending SIDER graph dataset preparation and curation].

### Definition 5 (Planned Cross-Axis Interaction Benchmark: Valid-Novel vs. Corrupted, $\mathcal{D}_{\text{interact}}$)
An evaluation task designed to test whether structural valency detection mechanisms remain reliable when applied to molecules that simultaneously exhibit scaffold novelty (corrupted graphs derived from or evaluated alongside novel chemotypes $\mathcal{D}_{\text{sem}}$). Planned evaluation task; exact test set assignment and corruption protocol: TODO: [Pending cross-axis split assignment].

---

## 4. Dataset Construction & Controlled Corruption

### 4.1 Tox21 Curation & Scaffold Splitting
We utilize the Tox21 toxicology benchmark comprising 12 bioassay targets. From an initial 7,831 compounds, 8 unparseable inorganic structures were filtered, yielding **7,823 validated molecules**. The dataset is partitioned via Bemis-Murcko scaffolds into a 70/10/10/10 scheme:
- `id_train` (5,476 molecules, ~70%): Primary GNN training set.
- `id_val` (782 molecules, ~10%): In-distribution validation and hyperparameter tuning.
- `id_test` (783 molecules, ~10%): In-distribution test set.
- `ood_valid` (782 molecules, ~10%): Axis 2 semantic OOD set with 100% disjoint scaffold cores.

### 4.2 Controlled Graph Corruption (Axis 1)
To generate non-trivial invalid graphs without altering overall atom compositions, we developed `add_bond_overvalent_carbon_v3`:
1. Identifies neutral, non-aromatic $sp^3/sp^2$ carbon pairs without an existing bond.
2. Inserts an explicit single covalent bond, driving valence to 5 ("Texas carbon").
3. Freezes explicit hydrogens to prevent automatic valence compensation.
4. Serializes topologies as raw graph JSON objects. **100% of corrupted graphs strictly fail RDKit chemical sanitization.**

### 4.3 Data Quality & Leakage Audit
- **Data Leakage:** Exact SMILES overlap = 0; scaffold hash overlap between ID and Axis 2 OOD = 0 (11/11 audit checks passed).
- **Shortcut Verification:** Mann-Whitney U tests and Cohen's $d$ across 6 molecular descriptors (molecular weight, atom count, bond count, TPSA, LogP, Rotatable Bonds) confirmed no shortcut separation (all $d < 0.11$, $p > 0.05$ for TPSA).

### 4.4 Planned External and Interaction Benchmarks
To evaluate extreme domain shift and multi-axis degradation, the research protocol includes two planned expansion tasks:
1. **ID vs. SIDER (Far-OOD Semantic Shift):** Benchmarking $\mathcal{D}_{\text{in}}$ against approved pharmaceutical molecules from SIDER. The exact dataset size, graph serialization, and target label filtering for SIDER are not yet established in the current repository: TODO: [Pending external SIDER dataset integration].
2. **Valid-Novel vs. Corrupted (Cross-Axis Interaction):** Assessing whether syntactic corruption detection is confounded by unfamiliar core scaffolds: TODO: [Pending interaction test partition generation].

---

## 5. Standardized Evaluation Protocol

Given ground truth binary label $y \in \{0, 1\}$ (0 for ID, 1 for OOD) and predicted anomaly score $S(G) \in \mathbb{R}$ (where larger scores indicate greater anomaly):

1. **AUROC (Area Under the Receiver Operating Characteristic):**
   $$\text{AUROC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$
2. **AUPR-Out ($\text{AUPR}_{\text{OOD}}$):** Area under the Precision-Recall curve with OOD treated as positive class ($y = 1$).
3. **AUPR-In ($\text{AUPR}_{\text{ID}}$):** Area under the Precision-Recall curve with ID treated as positive class ($y = 0$).
4. **FPR@95%TPR:** The False Positive Rate when the decision threshold $\tau$ achieves $\text{TPR}(\tau) \ge 0.95$:
   $$\text{FPR@95} = \text{FPR}(\tau), \quad \text{where } \tau = \sup \{ t : \text{TPR}(t) \ge 0.95 \}$$
5. **Combined Dual-Axis AUROC:**
   $$\text{AUROC}_{\text{mean}} = \frac{\text{AUROC}_{\text{axis1}} + \text{AUROC}_{\text{axis2}}}{2}$$
   $$\text{AUROC}_{\text{harmonic}} = \frac{2 \cdot \text{AUROC}_{\text{axis1}} \cdot \text{AUROC}_{\text{axis2}}}{\text{AUROC}_{\text{axis1}} + \text{AUROC}_{\text{axis2}}}$$

---

## 6. Experimental Results & Benchmarking

> **Note on Experimental Results:** In accordance with strict non-fabrication policies, all numerical benchmark scores below are marked as `TODO` pending completion of model training and evaluation runs by the experimental team across seeds 42, 123, and 456.

### Table 1: Dual-Axis OOD Detection Performance (Mean ± Std over 3 Seeds)

| Model | Detector | Axis 1 AUROC (Valency) | Axis 1 FPR95 (↓) | Axis 2 AUROC (Scaffold) | Axis 2 FPR95 (↓) | Combined Mean AUROC |
|---|---|---|---|---|---|---|
| Random Forest (ECFP4) | MSP | TODO: [Run EXP_001-006] | TODO | TODO | TODO | TODO |
| GCN | MSP | TODO: [Run EXP_007-012] | TODO | TODO | TODO | TODO |
| GCN | Energy | TODO: [Run EXP_013-018] | TODO | TODO | TODO | TODO |
| GIN | MSP | TODO: [Run EXP_019-024] | TODO | TODO | TODO | TODO |
| GIN | Energy | TODO: [Run EXP_025-030] | TODO | TODO | TODO | TODO |
| GAT | Mahalanobis | TODO: [Run EXP_031-036] | TODO | TODO | TODO | TODO |
| ChemBERTa | MSP | TODO: [Run EXP_037-042] | TODO | TODO | TODO | TODO |

---

## 7. Discussion & Chemical Failure Analysis

### 7.1 Can Standard Message Passing Detect Valency Violations?
TODO: [Analyze whether local permutation-invariant message passing without explicit valency inductive bias fails to detect pentavalent carbon bonds, based on experimental results].

### 7.2 Scaffold Generalization vs. Syntactic Detection Trade-offs
TODO: [Examine whether detectors optimized for semantic novelty (feature-space distance) generalize to topological syntax violations, or whether specialized hybrid detectors are required].

---

## 8. Limitations & Broader Impact

- **Dataset Scope:** Benchmarking is currently centered on Tox21 toxicological assays. Future expansions should evaluate kinase inhibition and quantum mechanical datasets (e.g., QM9).
- **Corruption Modality:** Structural corruptions focus on aliphatic overvalent carbon bonds. Future work should investigate hypervalent nitrogen and unphysical ring strain.
- **Safety Impact:** Reliable dual-axis detection prevents deployment of GNNs on chemically non-viable structures generated by de novo drug design algorithms.

---

## 9. Conclusion

We introduced the Dual-Axis OOD Detection benchmark for molecular graphs, decoupling physical valency violations from scaffold novelty. By establishing rigorous evaluation metrics and a reproducible, zero-leakage framework, this work provides a foundational stepping stone toward safer, more reliable molecular AI in drug discovery.

---

## References

1. Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. *Journal of Medicinal Chemistry*, 39(15), 2887-2893.
2. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR)*.
3. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *International Conference on Learning Representations (ICLR)*.
4. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *International Conference on Learning Representations (ICLR)*.
5. Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *International Conference on Learning Representations (ICLR)*.
6. Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based out-of-distribution detection. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 21464-21475.
7. Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *Advances in Neural Information Processing Systems (NeurIPS)*, 31.
8. Du, X., Wang, Z., Cai, M., & Li, Y. (2022). VOS: Learning what you don't know by virtual outlier synthesis. *International Conference on Learning Representations (ICLR)*.
9. Huang, W. et al. (2021). Tox21 Data Challenge: High-throughput screening for toxicology. *Frontiers in Environmental Science*, TODO: [Citation details].
10. TODO: [Additional domain-specific citations to be confirmed upon experimental completion].
