# Dual-Axis Out-of-Distribution (OOD) Detection — Project Analysis

---

## 1. Project Overview

This project prepares a dual-axis benchmark dataset for detecting **Out-of-Distribution (OOD)** molecular graphs. The term "dual-axis" refers to two orthogonal OOD axes that a future ML model must handle simultaneously:

| Axis | Name | What it tests |
|---|---|---|
| **Axis 1** | **Structural Validity** | Can the model detect *chemically invalid* molecular graphs (broken chemistry)? |
| **Axis 2** | **Semantic Novelty** | Can the model detect *chemically valid but structurally novel* molecules (unseen scaffolds)? |

The two axes are deliberately complementary — a model that only checks validity misses distribution shift; one that only checks novelty misses hard corruptions.

---

## 2. Source Dataset: Tox21

- **Source**: Tox21 dataset, fetched from the DeepChem S3 bucket (`tox21.csv.gz`)
- **Original size**: 7,831 molecules
- **Removed (invalid source)**: 8 molecules (aluminium-containing SMILES that fail RDKit sanitization — see `source_invalid_audit.csv`)
- **Working set**: **7,823 valid molecules**

### 12 Bioactivity Labels (multi-task)
`NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`, `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`

> [!NOTE]
> Labels have missing values (NaN), as is standard for Tox21. The split CSVs preserve all original label columns alongside the molecule metadata.

---

## 3. Dataset Splitting Strategy

All 7,823 valid molecules are partitioned into **four non-overlapping splits** using a **scaffold-based** strategy (Bemis–Murcko scaffolds via RDKit):

| Split | Count | % | Purpose |
|---|---|---|---|
| `id_train` | **5,476** | 70% | ID training data for model |
| `id_val` | **782** | 10% | ID validation (hyperparameter tuning) |
| `id_test` | **783** | 10% | ID test (held-out evaluation) |
| `ood_valid` | **782** | 10% | Scaffold-held-out OOD set (Axis 2) |

### Key Design Choices
- **Scaffold key**: Murcko scaffold SMILES for cyclic molecules. For **acyclic molecules**, the canonical SMILES is used as a fallback scaffold key (prefixed `ACYCLIC::`), since they have no ring system.
- **OOD split selection**: Entire scaffold *groups* are selected to meet the ~782 target, using a subset-sum greedy approach to ensure no scaffold leaks across splits.
- **Random seed**: `42` throughout for full reproducibility.
- **No canonical-SMILES overlap** between ID and OOD splits — scaffold isolation is strict.

---

## 4. Corruption Pipeline — Axis 1 (Structural Invalidity)

### Goal
Create chemically invalid molecular graphs paired with each ID molecule (train/val/test), to train the model to detect broken chemistry.

### Method: `add_bond_overvalent_carbon_v3`
1. Start with a valid RDKit molecule from the ID split.
2. Identify candidate carbon atoms: **non-aromatic, neutral carbons**.
3. Add a **single extra bond** between two such carbons (using a per-molecule deterministic seed `42000 + index`).
4. **Accept** only if the chosen carbon's explicit valence reaches **5** (over-valent; carbon normally supports max 4).
5. **Validate** that RDKit `SanitizeMol()` *fails* on the resulting graph (flag: `SANITIZE_PROPERTIES`).
6. Serialize the corrupted graph to JSON (`graph_json` field).

### Versioning History (in notebook)
| Version | Description |
|---|---|
| `v1` | Initial pilot — first attempt at bond corruption |
| `v2` | Refined: targeted saturated neutral carbons; added aromatic-safety checks |
| `v3` | **Final**: strict non-aromatic + neutral carbon filter; per-molecule deterministic seed; retained only confirmed over-valent corruptions |

### Corruption Results

| Split | Attempted | Successful | Failed | Success Rate |
|---|---|---|---|---|
| Train | 5,476 | **3,237** | 2,239 | 59.11% |
| Validation | 782 | **481** | 301 | 61.51% |
| Test | 783 | **477** | 306 | 60.92% |
| **Total** | **7,041** | **4,195** | **2,846** | **59.58%** |

> [!IMPORTANT]
> Failures (~40%) occur because some molecules have no suitable non-aromatic carbon that would become over-valent upon bond addition (e.g., fully aromatic molecules, molecules with only sp2 carbons). These failures are recorded in the audit files but **not** included in the invalid-graphs dataset.

### Graph JSON Schema (per invalid molecule)
```json
{
  "atoms": [
    {"atomic_num": 6, "formal_charge": 0, "aromatic": false,
     "chiral_tag": "CHI_UNSPECIFIED", "explicit_hs": 0, "no_implicit": false}
  ],
  "bonds": [
    {"begin": 0, "end": 1, "type": "SINGLE", "aromatic": false}
  ]
}
```

---

## 5. Reconstruction Validation

All 4,195 saved invalid graphs were **round-tripped** through JSON serialization → RDKit reconstruction → `SanitizeMol()`:

| Split | Total | Sanitization Failures | Reconstruction Errors | Failure Rate |
|---|---|---|---|---|
| train | 3,237 | 3,237 | 0 | 100% |
| val | 481 | 481 | 0 | 100% |
| test | 477 | 477 | 0 | 100% |

✅ **100% of reconstructed graphs failed sanitization** (as intended) — confirming no silent corruption during serialization/deserialization.

> [!NOTE]
> An intermediate step (Cell 26–27) found some graphs that "unexpectedly passed" reconstruction. These were investigated and traced to hydrogen-property serialization issues, which were fixed in the final `v3` pipeline (Cells 28–33) by preserving `explicit_hs` and `no_implicit` flags during `graph_to_json()`.

---

## 6. File Structure & Schema

### Valid Molecule Files
| File | Rows | Columns | Notes |
|---|---|---|---|
| `id_train.csv` | 5,476 | 18 | 12 labels + mol_id, smiles, rdkit_valid, canonical_smiles, scaffold_key, split |
| `id_val.csv` | 782 | 18 | Same schema |
| `id_test.csv` | 783 | 18 | Same schema |
| `ood_valid.csv` | 782 | 18 | Same schema; scaffold-held-out |
| `split_ids.csv` | 7,823 | 3 | mol_id, split, scaffold_key — master index |

### Invalid Graph Files (Axis 1 OOD)
| File | Rows | Key Columns |
|---|---|---|
| `invalid_graphs_train.csv` | 3,237 | source_mol_id, source_split, graph_json, atom_i, atom_j, explicit_valence_i, sanitization_flag |
| `invalid_graphs_val.csv` | 481 | Same |
| `invalid_graphs_test.csv` | 477 | Same |

### Audit & Validation Files
| File | Purpose |
|---|---|
| `corruption_audit_train/val/test.csv` | Full attempt log (success + failure rows) for all molecules |
| `rdkit_reconstruction_validation.csv` | Per-graph sanitization check results |
| `rdkit_validation_summary.csv` | Aggregate split-level summary |
| `source_invalid_audit.csv` | The 8 source molecules removed from Tox21 (AlH3-containing) |
| `dataset_manifest.json` | Machine-readable provenance: counts, seed, method, timestamp |

### Pilot Files
| File | Purpose |
|---|---|
| `corruption_pilot.csv` | v1 pilot run |
| `corruption_pilot_v2.csv` | v2 pilot run |
| `corruption_pilot_v3.csv` | v3 pilot run (final method validation) |

---

## 7. Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| **Scaffold-based OOD split** | Ensures OOD molecules have genuinely unseen ring systems, not just different substituents |
| **Acyclic fallback to canonical SMILES** | Prevents acyclic molecules from all clustering in one "no scaffold" group |
| **Over-valent carbon (valence=5)** | Unambiguous chemical invalidity; RDKit always rejects it; easy to verify |
| **Only non-aromatic carbons targeted** | Aromatic carbons handled differently by RDKit; skipping them avoids false positives |
| **Per-molecule deterministic seed** | Ensures exact reproducibility without global state |
| **JSON graph serialization** | Model-agnostic; works for GNN node/edge feature construction |
| **100% sanitization-failure check** | Guarantees all invalid graphs actually look invalid to downstream models |

---

## 8. Current Status & Limitations

### ✅ Completed
- Source data cleaning & 8-molecule audit
- Scaffold-based 4-way split (7,823 molecules)
- Invalid graph generation (4,195 graphs, 59.6% success rate)
- 100% reconstruction validation
- Full audit trail and reproducibility manifest

### ⚠️ Limitations / Open Questions
1. **~40% corruption failure rate**: Molecules without suitable non-aromatic carbons cannot be corrupted by this method — these molecules have no paired invalid graph in the training set.
2. **Single corruption type**: Only one structural invalidity type (over-valent carbon). Real-world chemical errors can be more diverse.
3. **Axis 2 not yet evaluated**: The `ood_valid` split is created but no scaffold-leakage verification audit has been run yet (noted as a next step in the summary).
4. **No model training yet**: This is a pure **data preparation** phase — no OOD detection model has been built yet.
5. **Label sparsity**: Tox21 has heavy NaN rates in bioactivity labels, which will affect multi-task learning.

---

## 9. Suggested Next Steps (as noted in the project summary)

1. **Scaffold-leakage audit**: Verify zero canonical-SMILES overlap between `id_*` and `ood_valid`.
2. **Model development**: Train a GNN (e.g., GIN, GAT) on `id_train` with the 12 Tox21 labels.
3. **Axis 1 OOD detector**: Build a classifier that distinguishes valid from invalid graphs using the `invalid_graphs_*.csv` files.
4. **Axis 2 OOD detector**: Use `ood_valid` for novelty detection (e.g., via embedding distance, energy-based models, or conformal prediction).
5. **Combined dual-axis evaluation**: Score models on both axes simultaneously.

---

## 10. Summary Statistics

```
Source dataset (Tox21):        7,831 original → 7,823 valid (8 removed)
ID Training molecules:         5,476   (70%)
ID Validation molecules:         782   (10%)
ID Test molecules:               783   (10%)
OOD Valid molecules:             782   (10%)

Corruption attempts:           7,041
Successful invalid graphs:     4,195   (59.58%)
Failed corruptions:            2,846   (40.42%)

Reconstruction validation:     4,195 / 4,195 failed sanitization (100%) ✅
Reconstruction errors:         0 ✅

Total output files:            18 (in Dataset/)
Random seed (all ops):         42
```
