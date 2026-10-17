# Step 9 - Final Scientific Audit, Robustness Review, and Paper-Readiness Gate

**Author:** Member 4 (Model Development & Evaluation)
**Date:** October 17, 2026
**Status:** COMPLETE - All Checks Passed
**Working Directory:** `step9_audit/`
**Artifact:** `step9_audit_summary.json`

---

## 1. Executive Summary

This is the **final technical audit** of the Dual-Axis OOD Detection project before paper writing begins.
All components from Steps 1-8 were independently verified under strictly frozen conditions.
No model weights, hyperparameters, or reported metrics were modified.

### Key Findings

| Check | Outcome |
|:---|:---:|
| Regression test suite (6/6 suites) | PASS |
| Checkpoint loadability (3/3 artifacts) | OK |
| Embedding dimension (all components) | 256d |
| Val metrics reproduction (7/7 metrics) | EXACT (Delta = 0.00e+00) |
| Test metrics reproduction (8/8 metrics) | EXACT (Delta = 0.00e+00) |
| Family C generalization (held-out) | AUROC = 0.9968 |
| Validity detector calibration (ECE) | ECE = 0.0215 |
| Mean score separation (id vs corrupt) | 0.8974 |
| Post-evaluation tuning | STRICTLY NONE |

All 15 independently recomputed metrics match the saved artifacts to bit-exact precision.
No data leakage, model drift, or post-hoc editing is detected.

---

## 2. Regression Test Suite (Step 9 Run)

| Test Suite | Tests | Exit Code | Outcome |
|:---|:---:|:---:|:---:|
| `tests/test_loader.py` | 4 | 0 | PASS |
| `tests/test_model.py` | 3 | 0 | PASS |
| `tests/test_validity.py` | 4 | 0 | PASS |
| `tests/test_novelty.py` | 6 | 0 | PASS |
| `tests/test_integration.py` | 6 | 0 | PASS |
| `tests/test_step7_novelty.py` | 5 | 0 | PASS |

All 6/6 suites pass. Consistent with the Step 8 regression gate.

---

## 3. Checkpoint Existence, Loadability, and Architecture Verification

### 3.1 Checkpoint Files

| Artifact | Path | Size | Status |
|:---|:---|:---:|:---:|
| Toxicity backbone | `checkpoints/toxicity_gine/best_model.pt` | 2,143 KB | OK |
| Validity detector | `checkpoints/validity_gine/best_validity_model.pt` | 2,072 KB | OK |
| Novelty estimator | `checkpoints/novelty_mahal/estimator.npz` | 479 KB | OK |

### 3.2 Embedding Dimension Verification

All three components independently loaded and dimension-checked:

- Toxicity GNN `embed_dim`: **256** OK
- Validity Detector `embed_dim`: **256** OK
- Novelty Estimator `embed_dim`: **256** OK

The 256-dimensional embedding (mean + sum pooling, 128d each) is confirmed consistent
across all frozen components. This was an open question from early Steps that is now
definitively resolved.

---

## 4. Independent Metric Reproduction

### 4.1 Validation Metrics (7/7 EXACT MATCH)

| Metric | Reproduced AUROC | Saved AUROC | Delta |
|:---|:---:|:---:|:---:|
| `validity_all` | 0.995896 | 0.995896 | 0.00e+00 |
| `validity_family_a` | 0.996583 | 0.996583 | 0.00e+00 |
| `validity_family_b` | 0.995207 | 0.995207 | 0.00e+00 |
| `novelty_scaffold_raw` | 0.456640 | 0.456640 | 0.00e+00 |
| `novelty_scaffold_matched` | 0.478300 | 0.478300 | 0.00e+00 |
| `novelty_dataset_raw` | 0.607449 | 0.607449 | 0.00e+00 |
| `novelty_dataset_matched` | 0.536174 | 0.536174 | 0.00e+00 |

### 4.2 Test Metrics (8/8 EXACT MATCH)

| Metric | Reproduced AUROC | Saved AUROC | Delta |
|:---|:---:|:---:|:---:|
| `validity_all_test` | 0.995834 | 0.995834 | 0.00e+00 |
| `validity_family_a_test` | 0.998600 | 0.998600 | 0.00e+00 |
| `validity_family_b_test` | 0.992083 | 0.992083 | 0.00e+00 |
| `validity_family_c_test` | 0.996819 | 0.996819 | 0.00e+00 |
| `novelty_scaffold_raw_test` | 0.501288 | 0.501288 | 0.00e+00 |
| `novelty_scaffold_matched_test` | 0.453556 | 0.453556 | 0.00e+00 |
| `novelty_dataset_raw_test` | 0.665997 | 0.665997 | 0.00e+00 |
| `novelty_dataset_matched_test` | 0.572965 | 0.572965 | 0.00e+00 |

Bit-exact reproducibility across both validation and test sets confirms fully deterministic
inference pipelines (seed=42, CPU-only).

---

## 5. Family C Generalization Verification

Family C (heavy-atom substitution, e.g., C -> O swaps) was strictly quarantined in `corrupt_test`
during all training and validation phases. The Step 9 audit evaluates it in a fresh inference run:

- Family C graphs in `corrupt_test`: **181**
- Family C AUROC (test): **0.9968**
- FPR95: **0.0000**

The GINE validity detector (trained entirely without Family C) generalizes near-perfectly
to this held-out corruption family. This provides strong evidence that the model learns
true valence constraints rather than family-specific memorization.

Family C was NEVER used for model selection, hyperparameter tuning, or checkpoint comparison.
This is the final confirmed result.

---

## 6. Validity Detector Calibration Analysis

### 6.1 Score Distributions

| Split | N | Mean Validity Score | Interpretation |
|:---|:---:|:---:|:---|
| `id_val` (in-distribution) | 679 | **0.0312** | Near-zero probability of corruption |
| `corrupt_val` (all families) | 543 | **0.9286** | Near-one probability of corruption |

Score separation: **0.8974** (mean corrupt minus mean ID)

### 6.2 Expected Calibration Error

**ECE (10-bin) = 0.0215**

| ECE Range | Assessment |
|:---|:---|
| ECE < 0.01 | Excellent calibration |
| **0.01 to 0.05** | **Good calibration (our result: 0.0215)** |
| 0.05 to 0.15 | Acceptable calibration |
| ECE > 0.15 | Poor calibration |

The validity detector is well-calibrated as a binary probability estimator.
Threshold-based deployment (score > 0.5 = predicted corrupted) is appropriate
without additional Platt scaling.

---

## 7. Consolidated Final Metrics

### 7.1 Axis 1 - Chemical Validity (All Verified in Step 9)

| Setting | AUROC | AUPR | FPR95 |
|:---|:---:|:---:|:---:|
| Validation: All Corruptions | 0.9959 | 0.9961 | 0.0044 |
| Validation: Family A (Valence) | 0.9966 | 0.9933 | 0.0103 |
| Validation: Family B (Bond) | 0.9952 | 0.9934 | 0.0044 |
| Test: All Corruptions | 0.9958 | 0.9965 | 0.0000 |
| Test: Family A | 0.9986 | 0.9961 | 0.0000 |
| Test: Family B | 0.9921 | 0.9885 | 0.0000 |
| **Test: Family C (Held-out)** | **0.9968** | 0.9942 | 0.0000 |

Atom-count baseline on Family A: AUROC = 0.5114 (validates GINE learns chemistry, not counting).

### 7.2 Axis 2 - Molecular Novelty (Verified Across Steps 5-8)

All novelty methods evaluated on identical matched controls from `processed_v3/tables/matched_controls.csv`.

| Method | Scaff Raw | Scaff Matched | Data Raw | Data Matched | Source |
|:---|:---:|:---:|:---:|:---:|:---:|
| GNN Mahalanobis (Step 5) | 0.4566 | 0.4783 | 0.6074 | 0.5362 | Step 9 Verified |
| Multi-Task Energy (Step 6) | 0.5653 | 0.5044 | 0.5281 | 0.5208 | Step 8 Verified |
| Latent kNN k=20 (Step 6) | 0.4958 | 0.4838 | 0.4982 | 0.5577 | Step 8 Verified |
| ECFP4 / Max Tanimoto (Step 7) | 0.5534 | 0.4950 | 0.4943 | 0.4652 | Step 8 Reproduced |
| GNN Mahalanobis (Test set) | 0.5013 | 0.4536 | 0.6660 | 0.5730 | Step 9 Verified |

Under matched controls, all novelty methods perform near chance (AUROC 0.46-0.55).
This is a structural property of the scaffold-disjoint benchmark, not a model failure.
It must be clearly and transparently communicated in the paper.

---

## 8. Statistical Robustness (from Step 8, 2000-replicate paired bootstrap)

| Comparison (Matched Controls) | Delta AUROC | 95% CI | p-value | Significant? |
|:---|:---:|:---:|:---:|:---:|
| ECFP4 vs GNN Mahalanobis (Scaffold) | +0.0167 | [-0.039, +0.072] | 0.559 | NO |
| ECFP4 vs Multi-Task Energy (Scaffold) | -0.0094 | [-0.071, +0.054] | 0.782 | NO |
| Multi-Task Energy vs GNN Mahal (Scaffold) | +0.0261 | [-0.034, +0.087] | 0.400 | NO |
| ECFP4 vs GNN Mahalanobis (Dataset) | -0.0710 | [-0.138, +0.002] | 0.056 | NO |

Under matched controls, no novelty method demonstrates a statistically significant advantage.
This null result is the scientifically honest and paper-ready conclusion.

---

## 9. Benchmark Design Insight (Key Finding for Paper)

### The Scaffold-Disjointness Problem

Because `processed_v3` uses Murcko scaffold-stratified splits:

- `id_val` scaffolds: 509 unique, **0% overlap** with `id_train`
- `novel_scaffold_val` scaffolds: 222 unique, **0% overlap** with `id_train`
- `id_val` scaffolds shared with `novel_scaffold_val`: **0 (0.0%)**

Any distance-to-training-set estimator assigns high novelty scores to BOTH `id_val` and
`novel_scaffold_val`, because both contain scaffolds unseen in training. Under size- and
ring-matched controls, this marginal separation collapses to chance-level AUROC.

### Generalization Beyond Our Benchmark

Scaffold-stratified splitting - the correct and standard approach for molecular ML - 
systematically undermines distance-to-training-set novelty detectors. Future work requires:

1. Novelty detectors conditioned on scaffold class
2. Pretraining objectives that align embedding distances with biological/chemical novelty
3. Benchmark designs that separately evaluate scaffold novelty vs property novelty

---

## 10. Paper Readiness Gate - Final Decision

| Condition | Requirement | Result | Status |
|:---|:---|:---:|:---:|
| Tests passing | 6/6 suites, exit 0 | 6/6 | PASS |
| Artifact integrity | 0 discrepancies | 0/15 | PASS |
| Family C holdout | AUROC > 0.90 | 0.9968 | PASS |
| Calibration | ECE < 0.05 | 0.0215 | PASS |
| No shortcut exploitation | Atom baseline AUROC < 0.60 on Fam A | 0.5114 | PASS |
| Statistical rigor | Paired bootstrap CIs reported | Complete | PASS |
| Matched controls | Coverage > 90% | 91-94% | PASS |
| No test-set tuning | Zero post-hoc tuning | Confirmed | PASS |
| Reproducibility | Bit-exact across runs | Confirmed | PASS |

**ALL GATE CONDITIONS SATISFIED. THE PROJECT IS PAPER-READY.**

---

## 11. Reproducibility Instructions

To reproduce the complete Step 9 audit from a clean shell:

```powershell
# 1. Verify environment
python -c "import torch, rdkit, sklearn, scipy, numpy, pandas; print('OK')"

# 2. Run regression test suite (6/6 suites)
python -c "
import subprocess, sys
tests = ['tests/test_loader.py', 'tests/test_model.py', 'tests/test_validity.py',
         'tests/test_novelty.py', 'tests/test_integration.py', 'tests/test_step7_novelty.py']
for t in tests:
    r = subprocess.run([sys.executable, t], capture_output=True, text=True, timeout=120)
    print(f'{\"PASS\" if r.returncode==0 else \"FAIL\"} {t}')
"

# 3. Run Step 9 audit (metric recomputation + calibration + Family C check)
python step9_audit/run_step9_audit.py
# Output: step9_audit/step9_audit_summary.json
```

---

## 12. Final Deliverables Summary

| Deliverable | Location | Status |
|:---|:---|:---:|
| Regression test results | Terminal output | 6/6 PASS |
| Step 9 audit runner | `step9_audit/run_step9_audit.py` | Done |
| Step 9 audit summary JSON | `step9_audit/step9_audit_summary.json` | Done |
| Step 9 final report | `step9_audit/STEP9_FINAL_AUDIT_REPORT.md` | Done |
| Frozen val predictions | `predictions/val_predictions.csv` | Preserved |
| Frozen test predictions | `predictions/test_predictions.csv` | Preserved |
| Frozen val metrics | `predictions/val_metrics.json` | Preserved |
| Frozen test metrics | `predictions/test_metrics.json` | Preserved |
| Step 8 reproducibility report | `step8_validation/STEP8_REPRODUCIBILITY_REPORT.md` | Preserved |
| Step 7 ECFP4 results | `step7_novelty/val_ecfp4_metrics.json` | Preserved |
| Step 6 audit results | `step6_audit/baseline_results.json` | Preserved |

---

*Audited by Member 4 (Model Development & Evaluation). All checks completed October 17, 2026.*
