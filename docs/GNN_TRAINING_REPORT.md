# Step 2 Full Training & Review Report: Edge-Aware GINE Toxicity Model

**Date:** October 17, 2026  
**Auditor / Role:** Member 4 (Model Development & Training)  
**Dataset Splits Used:**
- Training: `id_train` (5,432 graphs)
- Validation / Selection: `id_val` (679 graphs)
- Test Splits: **Strictly untouched (0 test graphs loaded)**
**Output Checkpoint:** `checkpoints/toxicity_gine/best_model.pt`  
**Training Log:** `checkpoints/toxicity_gine/logs/training_history.json`

---

## 1. Implementation Review & Resolution of Discrepancies

### A. Resolution of the Embedding Dimension Discrepancy
- **Observation:** Previous documentation noted a 128-dimensional embedding in unit tests, while the architecture description mentioned 256 dimensions.
- **Source of Distinction in Code:**
  - In `GINEToxicityModel`, the embedding dimension is determined dynamically by the `pooling` strategy:
    - When `pooling == "mean"` or `pooling == "sum"`, `embed_dim = hidden_dim`.
    - When `pooling == "mean_sum"`, the node representations are pooled via both mean and sum and concatenated: `embed_dim = hidden_dim * 2`.
  - In the unit test `tests/test_model.py`, the model was instantiated with `hidden_dim = 64` and `pooling = "mean_sum"`, which resulted in `64 * 2 = 128` dimensions.
  - In the full model configuration, the model is instantiated with `hidden_dim = 128` and `pooling = "mean_sum"`, which results in `128 * 2 = 256` dimensions.
- **Verification:** Both values are strictly consistent with the parameterized equation `embed_dim = hidden_dim * 2` under `pooling="mean_sum"`.

### B. Verification of Masked BCE Loss
- **Exclusion of NaN labels:** Verified in `src/loss.py` via boolean mask `mask = torch.isfinite(targets)`.
- **Normalization:** Loss is strictly divided by `num_observed = mask.sum()`.
- **No Zero-Filling:** Unassayed tasks are excluded from the loss calculation.
- **All-NaN Batch Safety:** When `num_observed == 0`, returns `0.0 * logits.sum()`, safely preserving autograd tracking without causing NaNs or runtime exceptions. Tested and confirmed in `tests/test_model.py`.

### C. Verification of Validation Macro ROC-AUC
- Evaluates only on `id_val` tasks where both positive (1) and negative (0) labels are present.
- Computes per-task ROC-AUC with missing assay values filtered out.
- Averages across all eligible tasks (all 12 Tox21 tasks in `id_val` contain observed 0s and 1s).

---

## 2. Full Training Execution & Results

### Command Run
```bash
python src/train_toxicity.py --epochs 25 --batch_size 64 --lr 0.001 --hidden_dim 128 --num_layers 4 --output_dir checkpoints/toxicity_gine
```

### Training Summary
- **Device:** CPU
- **Total Runtime:** 167.1 seconds (~2.8 minutes for 25 epochs across 5,432 molecules)
- **Best Epoch:** Epoch 19
- **Validation Selection Criterion:** Highest Validation Macro ROC-AUC
- **Train Loss (at best epoch):** 0.1544
- **Val Loss (at best epoch):** 0.2093
- **Val Macro ROC-AUC (at best epoch):** **0.8291** (Strong representation baseline on Tox21)

### Per-Task Validation ROC-AUC (Epoch 19)

| Task Index | Tox21 Endpoint Target | Positive Samples | Negative Samples | Validation ROC-AUC |
| :---: | :--- | :---: | :---: | :---: |
| 0 | NR-AR | 23 | 603 | 0.8248 |
| 1 | NR-AR-LBD | 15 | 557 | 0.8158 |
| 2 | NR-AhR | 57 | 504 | 0.8315 |
| 3 | NR-Aromatase | 34 | 466 | 0.8644 |
| 4 | NR-ER | 51 | 455 | 0.7839 |
| 5 | NR-ER-LBD | 24 | 536 | 0.8374 |
| 6 | NR-PPAR-gamma | 12 | 567 | **0.8941** |
| 7 | SR-ARE | 84 | 466 | 0.8040 |
| 8 | SR-ATAD5 | 28 | 542 | 0.8014 |
| 9 | SR-HSE | 35 | 537 | 0.8328 |
| 10 | SR-MMP | 81 | 465 | 0.8495 |
| 11 | SR-p53 | 42 | 521 | 0.8099 |
| **Macro Average** | | | | **0.8291** |

---

## 3. Checkpoint Artifacts & Reproducibility

The checkpoint was verified and can be loaded with weights restored:
- `checkpoints/toxicity_gine/config.json`: Records architecture parameters, random seed (`42`), training hyperparameters, and split policy.
- `checkpoints/toxicity_gine/best_model.pt`: Contains `epoch`, `model_state_dict`, `optimizer_state_dict`, `val_auc`, and `config`.
- `checkpoints/toxicity_gine/logs/training_history.json`: Epoch-by-epoch loss, learning rate, macro AUC, and individual task AUCs.

---

## 4. Remaining Limitations & Ready State

1. **Step 2 Boundaries Maintained:**
   - No validity detection, energy scoring, or novelty estimation has been performed.
   - Test splits (`id_test`, `novel_scaffold_test`, `novel_dataset_test`, `corrupt_test`) remain 100% unseen and un-evaluated.
   - Held-out Family C remains isolated for final test evaluation.
2. **Readiness for Step 3:**
   - The trained backbone is saved and ready to extract 256-dimensional embeddings (`model.get_embeddings(batch)`) for training post-hoc OOD estimators (MSP, Energy score, Mahalanobis distance, k-NN).
