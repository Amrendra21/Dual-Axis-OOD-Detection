# Summary Till EDA: Dual-Axis Out-of-Distribution (OOD) Detection

---

## 1. Project Overview & Motivation

### The Problem
In real-world drug discovery and molecular machine learning, Graph Neural Networks (GNNs) trained on known molecules are frequently deployed on molecules they have never seen before. When a model encounters inputs outside its training distribution, standard models make **confidently wrong predictions**.

Most research treats Out-of-Distribution (OOD) detection as a single binary problem. This project introduces **Dual-Axis OOD Detection**, splitting OOD behavior into two fundamentally distinct, orthogonal axes:

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

1. **Axis 1 — Structural Validity (Syntactic OOD):**
   - *Question:* Is this graph even a valid chemical molecule?
   - *Failure mode:* Does the graph violate fundamental physical/chemical valency rules (e.g., a neutral carbon with 5 covalent bonds — a "Texas carbon")?
   - *Why it matters:* Generative models or graph perturbation algorithms can generate graphs that violate valency rules. A chemical model must identify them as chemically non-viable.

2. **Axis 2 — Semantic / Scaffold Novelty (Semantic OOD):**
   - *Question:* Is this molecule chemically valid, but from an unseen core scaffold / structural class?
   - *Failure mode:* Model extrapolating molecular bioactivity onto totally unfamiliar chemical rings/cores.
   - *Why it matters:* Models trained on benzene- or steroid-heavy chemistries often fail silently on macrocycles or novel heterocyclic scaffolds.

---

## 2. The Raw Dataset (Tox21)

### Dataset Source
- **Dataset:** **Tox21** (Toxicology in the 21st Century), sourced from the NIH / EPA via DeepChem S3 (`tox21.csv.gz`).
- **Domain:** Bioassays measuring whether chemical compounds perturb 12 nuclear receptor (NR) and stress response (SR) pathways.
- **Initial Count:** 7,831 compounds.

### Data Cleaning Performed
- **SMILES Validation:** Parsed with RDKit's `Chem.MolFromSmiles`.
- **Failed Molecules:** Exactly **8 molecules failed RDKit parsing** (molecules containing aluminum hydride complexes like `[AlH3]`, which cannot be sanitized).
- **Final Cleaned Count:** **7,823 valid, unique molecules**.
- **Labels:** 12 binary classification targets (`NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`, `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`). Note that Tox21 labels have substantial missingness (NaNs), which is standard for high-throughput biological screening.

---

## 3. Data Processing & Graph Generation Pipeline

### Step 1: Bemis-Murcko Scaffold Splitting (70 / 10 / 10 / 10)
Rather than using random splitting (which causes data leakage because identical ring systems appear in both train and test sets), we split molecules by their **Bemis-Murcko Scaffold**:
- **Bemis-Murcko Scaffold:** Extracts the core ring systems and linkers, replacing all side-chain atoms/functional groups with generic linkers or single bonds.
- **Fallback for acyclic molecules:** Linear/branched molecules without rings (e.g., butane) have an empty scaffold string, so their canonical SMILES was used as the scaffold identifier.
- **Data Splitting Outcome:**
  - `id_train`: **5,476 molecules** (~70%) — In-distribution training set.
  - `id_val`: **782 molecules** (~10%) — In-distribution hyperparameter tuning.
  - `id_test`: **783 molecules** (~10%) — In-distribution evaluation.
  - `ood_valid`: **782 molecules** (~10%) — **Axis 2 (Semantic OOD)**: 100% disjoint scaffold cores never seen during training.

---

### Step 2: Controlled Graph Corruption (Axis 1: Syntactic OOD)
To construct the negative samples for Axis 1, we engineered a chemical graph corruption function: **`add_bond_overvalent_carbon_v3`**.

#### How the Corruption Works:
1. Parse the molecule without immediate strict sanitization.
2. Find all **neutral, non-aromatic carbon atoms** ($sp^3$, $sp^2$) in the molecule.
3. Check candidate pairs of carbons that do not already share a bond.
4. Add an **explicit single covalent bond** between two carbons, deliberately driving at least one carbon's total valence to **5** (or exceeding its maximum physical capacity of 4).
5. **Freeze explicit hydrogens** (`explicit_hs`) and set `no_implicit = True` so RDKit cannot "fix" the error by removing implicit hydrogens.
6. **Round-Trip Validation:** Reconstruct the molecule from JSON and run `Chem.SanitizeMol()`. If it throws a `KekulizeException` or `AtomValenceException`, the graph is verified as **strictly invalid**.

#### Corruption Statistics:
- **Target pool:** Evaluated across `id_train`, `id_val`, and `id_test` (7,041 total valid molecules).
- **Success count:** **4,195 invalid graphs generated** (~59.58% success rate).
- **Why ~40% fail:** Small molecules, fully aromatic compounds (e.g., pure benzenes/naphthalenes), or crowded structures lack two candidate aliphatic carbons where an extra bond can be added without violating RDKit's graph creation grammar.
- **Quality check:** **100% of the 4,195 exported invalid graphs fail RDKit sanitization**.

---

### Step 3: Graph Serialization Format
Standard chemical formats like SMILES **cannot represent invalid chemistry** (RDKit rejects them during string generation). Therefore, graphs were serialized directly into **JSON format**:
- **Atoms:** `atomic_num`, `formal_charge`, `is_aromatic`, `chiral_tag`, `explicit_hs`, `no_implicit`
- **Bonds:** `begin_atom_idx`, `end_atom_idx`, `bond_type`, `is_aromatic`
This preserves the exact adjacency matrix and atom attributes for PyTorch Geometric / DGL graphs.

---

## 4. Exploratory Data Analysis (EDA) & Validation Steps

We performed an in-depth Exploratory Data Analysis and audit to confirm data quality, prevent data leakage, and ensure the benchmark cannot be solved by cheap shortcuts:

### Step 1: Molecular Size & Complexity Distributions
- Analyzed: Heavy atom count, bond count, molecular weight, ring count, rotatable bonds, and Topological Polar Surface Area (TPSA).
- Visualized in `eda/plots/01_molecule_sizes.png`.
- **Result:** Median heavy atom count is **15.0 atoms** across all splits.

### Step 2: Atom & Bond Type Distribution
- Checked atom compositions (C, N, O, S, P, Halogens) and bond types (Single, Double, Triple, Aromatic).
- Visualized in `eda/plots/02_atom_type_distribution.png` and `eda/plots/03_bond_type_distribution.png`.
- **Result:** Atom frequencies match across `id_train`, `id_val`, `id_test`, and `ood_valid` (Carbon ~74%, Oxygen ~14%, Nitrogen ~8%). There is no chemical composition bias in the OOD split.

### Step 3: Scaffold Disjointness & Diversity
- Evaluated cumulative scaffold concentration.
- Visualized in `eda/plots/04_scaffold_distribution.png`.
- **Result:** High diversity — over 4,000 unique scaffolds across the 7,823 molecules.

### Step 4: Strict Leakage Audit (11 Checks)
- Audited using `eda/leakage_audit.csv` and visualized in `eda/plots/07_leakage_audit.png`.
- Checked:
  - SMILES overlap between all split pairs: **0 duplicates (PASSED)**
  - Scaffold overlap between ID splits and `ood_valid`: **0 overlap (PASSED)**
  - Molecule ID leakage between valid and invalid splits: **100% matched pairings without cross-split pollution (PASSED)**

### Step 5: Shortcut Analysis (Hypothesis Testing & Effect Size)
- *The Danger:* If `ood_valid` molecules were simply bigger or heavier than `id_train`, a simple decision rule like `if num_atoms > 30: return OOD` would score 99% AUROC without learning any graph representations.
- *Our Test:* Ran **Mann-Whitney U tests** and calculated **Cohen's $d$ effect sizes** comparing ID vs OOD across 6 descriptors.
- *Findings:*
  - Molecular Weight: Cohen's $d = 0.089$ (negligible)
  - Heavy Atom Count: Cohen's $d = 0.093$ (negligible)
  - Number of Rings: Cohen's $d = 0.108$ (negligible)
  - TPSA: $p = 0.793$, Cohen's $d = -0.010$ (no difference)
- *Conclusion:* **All Cohen's $d$ values are $< 0.11$** (well below the 0.20 threshold for a small effect). **No trivial shortcut exists!**

### Step 6: Valid vs. Invalid Graph Size Analysis
- Visualized in `eda/plots/09_valid_vs_invalid_size.png`.
- Invalid graphs have a mean size of 21.9 atoms vs 18.4 for the general pool.
- *Reason:* A natural selection effect — tiny molecules (like methane or ethanol) lack pairs of aliphatic carbons to accept an extra bond.

---

## 5. Important Project Files

| File / Path | Category | Purpose & Details |
|---|---|---|
| `data_preparation.ipynb` | Pipeline Code | Original 37-cell Jupyter notebook implementing downloading, cleaning, scaffold splitting, corruption v1-v3, and JSON graph export. |
| `Dataset/manifest.json` | Data Metadata | Complete catalog of the 18 generated CSV/JSON files, MD5 checksums, record counts, and split distributions. |
| `Dataset/id_train_clean.csv` | Primary Data | 5,476 clean in-distribution training molecules with 12 bioactivity targets and scaffold IDs. |
| `Dataset/id_val_clean.csv` | Primary Data | 782 in-distribution validation molecules. |
| `Dataset/id_test_clean.csv` | Primary Data | 783 in-distribution test molecules. |
| `Dataset/ood_valid_clean.csv` | Primary Data | 782 scaffold-disjoint molecules representing **Axis 2 (Semantic OOD)**. |
| `Dataset/*_corrupted_v3_graphs.json` | Primary Data | Serialized graph adjacency lists and atom features for **Axis 1 (Structural OOD)** invalid graphs (4,195 total). |
| `eda/EDA_Dual_Axis_OOD.ipynb` | EDA & Validation | Comprehensive 12-section notebook executing all statistical tests, audits, and plotting. |
| `eda/eda_analysis.py` | Reproducibility Script | Standalone Python CLI script that runs the entire EDA and generates reports and plots. |
| `eda/leakage_audit.csv` | Audit Report | Results table showing all 11 leakage checks passed with 0 violations. |
| `eda/shortcut_analysis.csv` | Statistical Evidence | U-statistic, p-values, and Cohen's $d$ metrics proving absence of size/property shortcuts. |
| `eda/evaluation_metrics.json` | Standards Doc | Standard metric definitions: **AUROC**, **AUPR-In/Out**, **FPR@95%TPR**, and **Macro-F1**. |
| `eda/data_quality_report.md` | Official Deliverable | Formal Data Quality & Validation Report signed off for the Data Quality Lead milestone. |
| `eda/plots/` (10 PNGs) | Visual Evidence | High-resolution publication plots (sizes, atoms, bonds, scaffolds, leakage, shortcuts). |

---

## 6. Viva Questions & Model Answers

### Question 1: What is Dual-Axis OOD Detection and why is a single OOD metric insufficient?
> **Answer:** Standard OOD detection combines all abnormal inputs into one bucket. However, in molecular science, an input can fail along two orthogonal dimensions:
> 1. **Axis 1 (Syntactic / Structural Validity):** The graph violates physical rules of chemistry (e.g., carbon with 5 bonds).
> 2. **Axis 2 (Semantic / Scaffold Novelty):** The graph is chemically valid, but its scaffold/core is novel compared to training data.
> A single metric cannot tell whether an algorithm is failing because of physical nonsense or domain extrapolation. Dual-axis detection separates syntactic validity from semantic generalization.

### Question 2: Why did you use Bemis-Murcko scaffold splitting instead of random splitting?
> **Answer:** Random splitting in molecular datasets leads to overly optimistic performance due to **analog bias** (or scaffold leakage). Two molecules can share the exact same active ring core (e.g., a steroid ring) while differing by only a single methyl group. If one is in train and the other in test, the model simply memorizes the core. 
> Bemis-Murcko splitting decomposes each molecule into its ring systems and linkers, grouping identical cores together. By assigning whole scaffold groups exclusively to either train or test, we measure true chemical generalization.

### Question 3: How did you implement graph corruption, and why call it a "Texas carbon"?
> **Answer:** In our `add_bond_overvalent_carbon_v3` function, we identify pairs of neutral, non-aromatic aliphatic carbon atoms that do not share a bond. We add an explicit covalent single bond between them, raising the explicit valence of at least one carbon from 4 to 5, and freeze explicit hydrogens to prevent RDKit from auto-adjusting valence.
> In organic chemistry, a pentavalent carbon is colloquially known as a "Texas carbon" (a reference to the "Lone Star" state having a 5-pointed star). Because carbon only has four valence orbitals ($2s, 2p_x, 2p_y, 2p_z$), a neutral 5-bond carbon is chemically impossible and immediately fails RDKit's `SanitizeMol()`.

### Question 4: Why did ~40% of molecules fail corruption? Is that a bug?
> **Answer:** No, it is not a bug; it is an expected chemical constraint. To create a clean, chemically controlled overvalent carbon without creating uninterpretable side effects:
> - We strictly target non-aromatic, neutral carbon pairs.
> - Molecules that are purely aromatic (e.g., benzene derivatives), very small (e.g., formaldehyde, ethanol), or structurally congested do not have two candidate carbons with available open valence.
> Rather than forcing messy multi-atom mutations, we strictly skip molecules that cannot yield a clean, single-bond overvalent modification.

### Question 5: Why did you store corrupted graphs in JSON rather than SMILES or SDF?
> **Answer:** SMILES strings rely on standard chemical valency rules during parsing and canonicalization. If you try to generate a SMILES for a pentavalent carbon, RDKit throws a sanitization error and refuses to export it. 
> To preserve the corrupted molecular graphs for machine learning input (e.g., PyTorch Geometric `Data(x, edge_index)`), we serialized the raw graph topology—nodes (atomic number, charge, explicit H count) and edges (adjacency pairs and bond types)—directly into JSON.

### Question 6: What is a "shortcut" in OOD detection, and how did you verify that your dataset doesn't have one?
> **Answer:** A shortcut occurs when a nuisance attribute (such as molecule size or molecular weight) correlates strongly with the OOD label. If OOD molecules were significantly larger than ID molecules, a trivial threshold on atom count would yield 99% AUROC without the model learning any chemical topology.
> We verified this using **Mann-Whitney U non-parametric hypothesis testing** and **Cohen's $d$ effect size analysis** across 6 key descriptors (MW, Atom Count, Bond Count, Rings, Rotatable Bonds, TPSA). All Cohen's $d$ values were **$< 0.11$** (well below the threshold for even a small effect of 0.20), proving that our splits cannot be separated by simple size or property shortcuts.

### Question 7: What evaluation metrics should be used for evaluating the models, and why not just accuracy?
> **Answer:** In OOD detection, accuracy depends on the chosen classification threshold and is sensitive to class imbalance. We standardized three primary metrics:
> 1. **AUROC (Area Under the ROC Curve):** Measures discrimination ability across all thresholds (independent of class balance).
> 2. **AUPR (Area Under the Precision-Recall Curve, both In and Out):** Critical because OOD detection is typically a low-prevalence/imbalanced scenario.
> 3. **FPR@95%TPR (False Positive Rate at 95% True Positive Rate):** Measures how many valid ID samples are misclassified as anomalous when detecting 95% of true OOD cases (the gold standard for safety-critical deployment).

### Question 8: How did you ensure there was zero data leakage?
> **Answer:** We ran an automated 11-point audit checking:
> 1. Zero SMILES string overlap across all train, validation, test, and OOD sets.
> 2. Zero scaffold hash overlap between ID sets and `ood_valid`.
> 3. Strict 1-to-1 mapping of parent molecule IDs to corrupted graph IDs to guarantee that no test-corrupted graph originated from an ID-train molecule.
> All 11 checks returned 0 violations.
