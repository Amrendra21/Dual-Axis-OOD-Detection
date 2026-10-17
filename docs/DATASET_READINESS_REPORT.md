# Dataset Readiness Report: processed_v3 & Data Loader Verification

**Date:** October 17, 2026  
**Auditor / Role:** Member 4 (Model Development & Training)  
**Dataset Target:** `processed_v3`  
**Loader Implementation:** `src/data.py`  
**Smoke Test Suite:** `tests/test_loader.py`  
**Overall Readiness Status:** **READY FOR MODEL TRAINING** (Exit Code 0)

---

## 1. Executive Summary

This report documents the verification of the `processed_v3` dataset and the implementation of a PyTorch dataset loader and batching module (`src/data.py`).

Every one of the **10,140 molecular graphs** across all 10 `.pt` files was exhaustively inspected for tensor dimensions, finite values, connectivity integrity, and missing-label semantics. Zero schema errors and zero duplicate UIDs were found. Split integrity and twin pairing relationships were cross-verified against `split_ids.json`, `master_table.csv`, and `eligible_pairs.csv`, confirming zero cross-split leakage.

The newly implemented loader was validated via an automated smoke test (`tests/test_loader.py`), passing single-graph loading, disjoint union batch collation, UID recovery, missing label preservation (NaN), and complete split coverage.

---

## 2. Dataset Paths & Files Inspected

### Graph Tensor Files (`processed_v3/graphs/`)
- Total files: 10
- Inspection method: **Exhaustive (all 10,140 graphs inspected)**

| Split File | Count | Node Feature (`x`) | Edge Index (`edge_index`) | Edge Attribute (`edge_attr`) | Labels (`y`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `id_train.pt` | 5,432 | `[N, 30]`, float32 | `[2, E]`, int64 | `[E, 6]`, float32 | `[1, 12]`, float32 |
| `id_val.pt` | 679 | `[N, 30]`, float32 | `[2, E]`, int64 | `[E, 6]`, float32 | `[1, 12]`, float32 |
| `id_test.pt` | 679 | `[N, 30]`, float32 | `[2, E]`, int64 | `[E, 6]`, float32 | `[1, 12]`, float32 |
| `novel_scaffold_val.pt` | 408 | `[N, 30]`, float32 | `[2, E]`, int64 | `[E, 6]`, float32 | `[1, 12]`, float32 |
| `novel_scaffold_test.pt` | 347 | `[N, 30]`, float32 | `[2, E]`, int64 | `[E, 6]`, float32 | `[1, 12]`, float32 |
| `novel_dataset_val.pt` | 225 | `[N, 30]`, float32 | `[2, E]`, int64 | `[E, 6]`, float32 | `[1, 12]`, float32 |
| `novel_dataset_test.pt` | 198 | `[N, 30]`, float32 | `[2, E]`, int64 | `[E, 6]`, float32 | `[1, 12]`, float32 |
| `corrupt_train.pt` | 1,086 | `[N, 30]`, float32 | `[2, E]`, int64 | `[E, 6]`, float32 | `[1, 12]`, float32 |
| `corrupt_val.pt` | 543 | `[N, 30]`, float32 | `[2, E]`, int64 | `[E, 6]`, float32 | `[1, 12]`, float32 |
| `corrupt_test.pt` | 543 | `[N, 30]`, float32 | `[2, E]`, int64 | `[E, 6]`, float32 | `[1, 12]`, float32 |
| **Total** | **10,140** | | | | |

### Metadata and Tables Inspected
- `processed_v3/config.json`: Build args, seed `42`, platform/versions, SHA-256 hashes.
- `processed_v3/feature_schema.json`: 30 atom features, 6 bond features, 12 Tox21 tasks.
- `processed_v3/splits/split_ids.json`: UID lists for all 10 splits.
- `processed_v3/tables/master_table.csv`: 10,140 rows $\times$ 33 columns.
- `processed_v3/tables/eligible_pairs.csv`: 2,172 corrupt-to-twin pairs.
- `processed_v3/tables/matched_controls.csv`: 1,178 novelty-to-ID matched controls.

---

## 3. Schema & Tensor Verification Findings

1. **UID Preservation:**
   - 100% of graphs contain a valid string `uid` (e.g., `T21-00000`, `SID-00001`, `COR-A-T21-00005`).
   - Exactly 0 duplicate UIDs across the entire repository.
2. **Feature Dimensions:**
   - Node features `x`: shape `[N, 30]`, `torch.float32`.
   - Edge index `edge_index`: shape `[2, E]`, `torch.int64`, symmetric bidirectional.
   - Edge attributes `edge_attr`: shape `[E, 6]`, `torch.float32`.
   - Labels `y`: shape `[1, 12]`, `torch.float32`.
   - Matched `feature_schema.json` with 100% precision.
3. **Finite Values:**
   - Zero `NaN`, `Inf`, or `-Inf` in `x` or `edge_attr`.
   - All edge indices satisfy $0 \le \text{edge\_index}_{i,j} < N$.
4. **Missing Label Semantics:**
   - In-distribution and novel scaffold splits contain real Tox21 assay measurements alongside missing values represented strictly as `torch.nan`.
   - Corrupted graphs and SIDER (`novel_dataset`) graphs are 100% `torch.nan` for `y`.
   - Missing labels are **never** replaced with zeros or treated as negative examples.

---

## 4. Split Integrity & Leakage Safeguards

1. **Split Consistency:**
   - Counts match exactly across `.pt` files, `split_ids.json`, and `master_table.csv`.
   - Zero molecule overlap across splits.
2. **Twin Pairing Verification (`eligible_pairs.csv`):**
   - All 2,172 corrupted graphs map to a valid twin located strictly in the same split partition (`train` $\to$ `train`, `val` $\to$ `val`, `test` $\to$ `test`).
   - Cross-split twin leakage count: **0**.
3. **Family C Isolation:**
   - Family C (Carbon $\to$ Oxygen swap) exists **only** in `corrupt_test` ($n=181$).
   - Confirmed absent from `corrupt_train` and `corrupt_val`.

---

## 5. Implementation Summary

### Files Created
1. `src/data.py`:
   - `MolecularDataset(Dataset)`: Reads individual `.pt` files safely via `torch.load(..., weights_only=True)`.
   - `BatchGraphs`: Dataclass container representing batched graphs via disjoint union (concatenated `x`, shifted `edge_index`, concatenated `edge_attr`, stacked `y` `[B, 12]`, `batch` assignment vector `[total_nodes]`, `ptr` node offset vector `[B+1]`, and recovered `uid` list).
   - `collate_molecular_graphs`: Collation function for standard PyTorch `DataLoader`.
   - `get_dataloader`: Factory function to instantiate data loaders with configurable `batch_size`, `shuffle`, and `num_workers`.
2. `src/__init__.py`: Package exports for clean imports.
3. `tests/test_loader.py`: Automated smoke test verifying single graph loading, batch collation, UID recovery, NaN retention, and 10-split coverage.

---

## 6. How to Reproduce the Smoke Test

Execute from repository root:

```bash
python tests/test_loader.py
```

### Verified Test Output
```text
============================================================
RUNNING DATASET LOADER SMOKE TEST
============================================================

--- Test 1: Load single graph from id_train ---
Dataset id_train size: 5432 graphs
Graph 0 UID        : T21-00000
Graph 0 x shape    : torch.Size([16, 30]) (dtype: torch.float32)
Graph 0 edge_index : torch.Size([2, 34]) (dtype: torch.int64)
Graph 0 edge_attr  : torch.Size([34, 6]) (dtype: torch.float32)
Graph 0 y shape    : torch.Size([1, 12]) (dtype: torch.float32)
PASSED: Single graph schema verified.

--- Test 2: Batch loading via DataLoader ---
Batch num_graphs   : 5
Batch UIDs         : ['T21-00000', 'T21-00001', 'T21-00003', 'T21-00004', 'T21-00005']
Batch x shape      : torch.Size([82, 30])
Batch edge_index   : torch.Size([2, 164])
Batch edge_attr    : torch.Size([164, 6])
Batch y shape      : torch.Size([5, 12])
Batch vector shape : torch.Size([82])
Batch ptr vector   : [0, 16, 31, 51, 62, 82]
PASSED: Batch collation and UID recovery verified.

--- Test 3: Missing labels remain NaN ---
NaN count in batch labels: 7 / 60
Corrupted graphs labels NaN status: 100% NaN verified (48 NaNs)
SIDER graphs labels NaN status: 100% NaN verified (48 NaNs)
PASSED: Missing label semantics strictly preserved without zero-filling.

--- Test 4: Verify loader on all 10 splits without leakage ---
  Checked split id_train                 : 5432 graphs OK
  Checked split id_val                   : 679 graphs OK
  Checked split id_test                  : 679 graphs OK
  Checked split novel_scaffold_val       : 408 graphs OK
  Checked split novel_scaffold_test      : 347 graphs OK
  Checked split novel_dataset_val        : 225 graphs OK
  Checked split novel_dataset_test       : 198 graphs OK
  Checked split corrupt_train            : 1086 graphs OK
  Checked split corrupt_val              : 543 graphs OK
  Checked split corrupt_test             : 543 graphs OK
PASSED: All 10 splits loaded with exact expected counts.

============================================================
ALL SMOKE CHECKS PASSED SUCCESSFULLY (Exit Code 0)
============================================================
```

---

## 7. Discrepancies & Recommendations for Step 2

1. **Constant Features:**
   Features `degree_0`, `num_h_4plus`, and `bond_other` are constant zeros across the entire dataset. Models must not rely on variance from these columns.
2. **Multi-Task Masked Loss Required:**
   Tox21 training labels contain ~15.9% missing entries (`NaN`). When implementing model training (Step 2), a masked binary cross-entropy loss must be used:
   $$\mathcal{L} = \frac{\sum_{i,j} \mathbb{I}(y_{i,j} \ne \text{NaN}) \cdot \ell_{\text{BCE}}(\hat{y}_{i,j}, y_{i,j})}{\sum_{i,j} \mathbb{I}(y_{i,j} \ne \text{NaN})}$$
3. **Environment Compatibility:**
   `torch` 2.11.0+cpu is available. The custom `BatchGraphs` implementation in `src/data.py` operates directly with standard PyTorch tensors without requiring native `torch_geometric` C++ binary extensions, ensuring portability.
