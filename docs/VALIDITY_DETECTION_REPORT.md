# Step 3 Report: Separate GINE Validity Detector Implementation

**Date:** October 17, 2026  
**Auditor / Role:** Member 4 (Model Development & Training)  
**Dataset Target:** `processed_v3`  
**Base Toxicity Checkpoint:** `checkpoints/toxicity_gine/best_model.pt`  
**Trained Validity Checkpoint:** `checkpoints/validity_gine/best_validity_model.pt`  
**Training History Log:** `checkpoints/validity_gine/logs/training_history.json`  
**Test Suite:** `tests/test_validity.py`  

---

## 1. Executive Summary & Objective

In this step, we implemented a dedicated **structural validity detector** (`GINEValidityDetector`) to distinguish chemically impossible molecular graphs from valid molecules.

Key achievements:
1. **Explicit Task Formulation:** Valid molecules are the negative class ($y=0.0$), corrupted molecules are the positive class ($y=1.0$). The output `validity_score` is normalized to $[0.0, 1.0]$, where **higher means more corrupted / OOD**.
2. **Architecture & Representation:** Reuses the edge-aware GINE message-passing backbone (incorporating 30-dim atom features and 6-dim bond features) to capture valence and topological constraints, followed by a dedicated binary classification head.
3. **Strict Holdout of Family C:** Family C (held-out Carbon $\to$ Oxygen swap) exists exclusively in `corrupt_test` and is strictly excluded from all training, validation, and model selection.
4. **Validation Breakdown:** Model selection is driven by **Family A AUROC** (the gold standard count-invariant valence test), with Family B reported separately.
5. **Results on Validation Set:** The trained detector achieves **0.9966 AUROC on Family A** and **0.9952 AUROC on Family B** on `corrupt_val` vs. `id_val`.

---

## 2. Verification of Prerequisites

1. **Existing Checkpoint & Embeddings:**
   - Loaded `checkpoints/toxicity_gine/best_model.pt`.
   - Verified that `model.get_embeddings(batch)` produces finite 256-dimensional embeddings for batches from `id_train` and `id_val`.
2. **Metadata & Split Verification:**
   - Authoritative counts verified in `processed_v3/tables/master_table.csv`:
     - `id_train`: 5,432 valid graphs.
     - `corrupt_train`: 1,086 corrupted graphs (Family A: 543, Family B: 543, Family C: 0).
     - `id_val`: 679 valid graphs.
     - `corrupt_val`: 543 corrupted graphs (Family A: 272, Family B: 271, Family C: 0).
     - `corrupt_test`: 543 graphs (Family A: 181, Family B: 181, Family C: 181).
   - Zero Family C examples exist in train or validation sets.
3. **Pre-change Regression Tests:**
   - `python tests/test_loader.py` and `python tests/test_model.py` passed with exit code 0 prior to any changes.

---

## 3. Strict Split & Holdout Policy

```
[TRAINING PHASE]
  ├── id_train (5,432 graphs) ─────────── Label 0.0 (Valid)
  └── corrupt_train (1,086 graphs) ────── Label 1.0 (Corrupted: Fam A & Fam B only)

[VALIDATION / SELECTION PHASE]
  ├── id_val (679 graphs) ─────────────── Negative baseline
  └── corrupt_val (543 graphs) ────────── Positive validation:
                                          ├── Family A (n=272) -> Selection Metric
                                          └── Family B (n=271) -> Tracked separately

[TEST PHASE (FROZEN / UNSEEN)]
  ├── id_test (679 graphs) ────────────── UNTOUCHED
  ├── corrupt_test (543 graphs) ───────── UNTOUCHED
  │     ├── Family A (n=181) ──────────── UNTOUCHED
  │     ├── Family B (n=181) ──────────── UNTOUCHED
  │     └── Family C (n=181) ──────────── STRICTLY HELD OUT
```

Enforced by code in `ValidityDataset`: attempting to instantiate a dataset with `corrupt_test` raises a `ValueError` immediately.

---

## 4. Architecture & Training Details

### Model Specification (`src/validity_model.py`)
- **Encoder:** Reuses the 4-layer edge-aware GINE architecture initialized from `checkpoints/toxicity_gine/best_model.pt`.
- **Validity Head:** Separate 2-layer MLP (`Linear(256, 64) -> ReLU -> Dropout(0.1) -> Linear(64, 1)`).
- **Fine-Tuning:** The encoder layers are fine-tuned jointly with the validity head on `id_train + corrupt_train` using a lower learning rate ($10^{-3}$) and weight decay ($10^{-5}$).
- **Class Imbalance Handling:** Positive weight `pos_weight = 5432 / 1086 = 5.0` applied to `nn.BCEWithLogitsLoss`.
- **Output API:** `detector.score_batch(batch)` outputs probabilities in $[0.0, 1.0]$ in the exact order of `batch.uid`.

---

## 5. Training Results & Validation Breakdown

### Execution Command
```bash
python src/train_validity.py --epochs 6 --batch_size 64 --lr 0.001 --output_dir checkpoints/validity_gine
```

### Metrics Across Epochs
- **Runtime:** 95.5 seconds (CPU)
- **Best Epoch:** Epoch 6
- **Selection Criterion:** Highest Validation Family A AUROC

| Epoch | Train Loss | Val AUROC (All) | Val AUROC (Family A) | Val AUROC (Family B) | Best Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.4435 | 0.7961 | 0.7291 | 0.8634 | Saved |
| 2 | 0.5757 | 0.9792 | 0.9784 | 0.9800 | Saved |
| 3 | 0.2829 | 0.9840 | 0.9785 | 0.9896 | Saved |
| 4 | 0.1485 | 0.9961 | 0.9952 | 0.9969 | Saved |
| 5 | 0.1937 | 0.9932 | 0.9902 | 0.9962 | - |
| 6 | 0.1275 | **0.9959** | **0.9966** | **0.9952** | **Best Saved** |

### Significance of Family A Performance
In `audits/corruption_validation_report.md`, Family A (bond inflation) was proven to leave all atom counts, bond counts, degrees, and element counts **100% identical** between valid and corrupted twins (simple count models achieved an AUROC of ~0.49–0.51).
The GINE validity detector achieves an **AUROC of 0.9966 on Family A**, confirming that edge-aware message passing successfully learns relational chemical valence constraints rather than superficial counting shortcuts.

---

## 6. Unit & Integration Tests

Execute full test suite:
```bash
python tests/test_validity.py
```

### Verified Checks (All Passed)
1. **Dataset & Holdout Protection:** Verified loading of 6,518 training graphs; confirmed `ValueError` is raised if test splits are passed.
2. **Forward Pass & Score Properties:** Output tensor has shape `[batch_size]`, values in $[0.0, 1.0]$, and strictly matches the order of `batch.uid`.
3. **Score Direction:** Verified that corrupted graphs receive higher scores than valid graphs.
4. **Checkpoint Restoration:** Verified that weights saved to disk reload identically.

---

## 7. Deliverables Summary

| File | Purpose |
| :--- | :--- |
| [src/validity_model.py](file:///c:/Users/aryan/OneDrive/Documents/PROJECTS/OOD_Research/Dual-Axis-OOD-Detection/src/validity_model.py) | `GINEValidityDetector`, `ValidityDataset`, and batch collation. |
| [src/train_validity.py](file:///c:/Users/aryan/OneDrive/Documents/PROJECTS/OOD_Research/Dual-Axis-OOD-Detection/src/train_validity.py) | Full training loop, family-decoupled evaluation, and smoke test CLI. |
| [tests/test_validity.py](file:///c:/Users/aryan/OneDrive/Documents/PROJECTS/OOD_Research/Dual-Axis-OOD-Detection/tests/test_validity.py) | Unit tests verifying holdout enforcement, score direction, and shapes. |
| [checkpoints/validity_gine/best_validity_model.pt](file:///c:/Users/aryan/OneDrive/Documents/PROJECTS/OOD_Research/Dual-Axis-OOD-Detection/checkpoints/validity_gine/best_validity_model.pt) | Saved best checkpoint (Val Fam A AUROC: 0.9966). |
| [checkpoints/validity_gine/config.json](file:///c:/Users/aryan/OneDrive/Documents/PROJECTS/OOD_Research/Dual-Axis-OOD-Detection/checkpoints/validity_gine/config.json) | Reproducible configuration and split declaration. |
| [checkpoints/validity_gine/logs/training_history.json](file:///c:/Users/aryan/OneDrive/Documents/PROJECTS/OOD_Research/Dual-Axis-OOD-Detection/checkpoints/validity_gine/logs/training_history.json) | Detailed per-epoch loss and family-wise validation AUROCs. |
| [docs/VALIDITY_DETECTION_REPORT.md](file:///c:/Users/aryan/OneDrive/Documents/PROJECTS/OOD_Research/Dual-Axis-OOD-Detection/docs/VALIDITY_DETECTION_REPORT.md) | Comprehensive documentation of the validity detection step. |

Step 3 is complete and verified. The validity detector is frozen and ready for integration into the dual-axis evaluation framework.
