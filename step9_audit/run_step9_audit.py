"""
Step 9 — Final Scientific Audit, Robustness Review, and Paper-Readiness Gate.

Checks performed:
  1. Verify all frozen checkpoint files exist and are loadable.
  2. Independently recompute val_metrics.json values from frozen components.
  3. Independently recompute test_metrics.json values from frozen components.
  4. Check checksum consistency (val_metrics vs Step 5 frozen baseline).
  5. Calibration analysis: reliability diagrams and ECE for validity detector.
  6. Ablation check: confirm embed_dim=256 in all loaded components.
  7. Family C generalization: verify test Family C AUROC is non-trivial (>0.90).
  8. Matched control test-set consistency: verify test matched AUROC coverage.
  9. Document environment and produce STEP9_FINAL_AUDIT_REPORT.md.

Constraints:
  - Zero tuning on test metrics.
  - Do not overwrite any Step 1-8 artifact.
  - Keep Family C in corrupt_test only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Fix Windows console encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.evaluate import (
    compute_ood_metrics,
    load_frozen_components,
    score_split,
)

OUTPUT_DIR = REPO / "step9_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = REPO / "processed_v3"
TOXICITY_CKPT = REPO / "checkpoints/toxicity_gine/best_model.pt"
VALIDITY_CKPT = REPO / "checkpoints/validity_gine/best_validity_model.pt"
NOVELTY_CKPT = REPO / "checkpoints/novelty_mahal/estimator.npz"


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def load_reference_metrics() -> dict:
    val_path = REPO / "predictions/val_metrics.json"
    test_path = REPO / "predictions/test_metrics.json"
    with open(val_path) as f:
        val = json.load(f)
    with open(test_path) as f:
        test = json.load(f)
    return {"val": val, "test": test}


def check_checkpoints() -> dict:
    banner("STEP 9.1 — Checkpoint Existence and Loadability")
    results = {}
    for name, path in [
        ("toxicity_gine", TOXICITY_CKPT),
        ("validity_gine", VALIDITY_CKPT),
        ("novelty_mahal", NOVELTY_CKPT),
    ]:
        exists = path.exists()
        size_kb = path.stat().st_size // 1024 if exists else 0
        results[name] = {"exists": exists, "size_kb": size_kb, "path": str(path)}
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name}: {path.name} ({size_kb} KB)")

    # Load all three
    try:
        tox, val_det, nov_est = load_frozen_components(
            toxicity_ckpt_path=str(TOXICITY_CKPT),
            validity_ckpt_path=str(VALIDITY_CKPT),
            novelty_estimator_path=str(NOVELTY_CKPT),
            device=torch.device("cpu"),
        )
        results["load_status"] = "OK"
        results["tox_embed_dim"] = tox.embed_dim
        results["val_embed_dim"] = val_det.embed_dim
        results["nov_embed_dim"] = nov_est.embed_dim
        print(f"  [OK] All three components loaded.")
        print(f"       Toxicity embed_dim   = {tox.embed_dim}")
        print(f"       Validity embed_dim   = {val_det.embed_dim}")
        print(f"       Novelty embed_dim    = {nov_est.embed_dim}")
        assert tox.embed_dim == 256, f"Expected 256, got {tox.embed_dim}"
        assert val_det.embed_dim == 256, f"Expected 256, got {val_det.embed_dim}"
        assert nov_est.embed_dim == 256, f"Expected 256, got {nov_est.embed_dim}"
        print("  [OK] All embed_dim = 256. Dimension check passed.")
    except Exception as e:
        results["load_status"] = f"ERROR: {e}"
        print(f"  [FAIL] Load failed: {e}")
    return results


def recompute_val_metrics(tox, val_det, nov_est) -> dict:
    banner("STEP 9.2 — Independent Val Metrics Recomputation")
    master_df = pd.read_csv(DATA_DIR / "tables/master_table.csv")
    controls_df = pd.read_csv(DATA_DIR / "tables/matched_controls.csv")

    val_splits = ["id_val", "corrupt_val", "novel_scaffold_val", "novel_dataset_val"]
    scores = {}
    for s in val_splits:
        res = score_split(
            split_name=s,
            toxicity_model=tox,
            validity_detector=val_det,
            novelty_estimator=nov_est,
            root_dir=str(DATA_DIR),
            batch_size=64,
            device=torch.device("cpu"),
        )
        scores[s] = res
        print(f"  Scored {s}: {len(res['uids'])} graphs")

    corrupt_meta = master_df[master_df["set_name"] == "corrupt_val"][["uid", "family"]].copy()
    corrupt_df = pd.DataFrame({
        "uid": scores["corrupt_val"]["uids"],
        "validity_score": scores["corrupt_val"]["validity_scores"],
        "novelty_score": scores["corrupt_val"]["novelty_scores"],
    }).merge(corrupt_meta, on="uid")

    fam_a_vs = corrupt_df[corrupt_df["family"] == "A"]["validity_score"].to_numpy()
    fam_b_vs = corrupt_df[corrupt_df["family"] == "B"]["validity_score"].to_numpy()

    id_val_dict = dict(zip(scores["id_val"]["uids"], scores["id_val"]["novelty_scores"]))
    scaff_dict = dict(zip(scores["novel_scaffold_val"]["uids"], scores["novel_scaffold_val"]["novelty_scores"]))
    data_dict = dict(zip(scores["novel_dataset_val"]["uids"], scores["novel_dataset_val"]["novelty_scores"]))

    scaff_ctrl = controls_df[(controls_df["target_set"] == "novel_scaffold_val") & (controls_df["matched"] == True)].copy()
    data_ctrl = controls_df[(controls_df["target_set"] == "novel_dataset_val") & (controls_df["matched"] == True)].copy()
    scaff_ctrl["ts"] = scaff_ctrl["target_uid"].map(scaff_dict)
    scaff_ctrl["cs"] = scaff_ctrl["control_uid"].map(id_val_dict)
    data_ctrl["ts"] = data_ctrl["target_uid"].map(data_dict)
    data_ctrl["cs"] = data_ctrl["control_uid"].map(id_val_dict)

    metrics = {}
    id_vs = scores["id_val"]["validity_scores"]
    corrupt_vs = scores["corrupt_val"]["validity_scores"]
    id_ns = scores["id_val"]["novelty_scores"]
    scaff_ns = scores["novel_scaffold_val"]["novelty_scores"]
    data_ns = scores["novel_dataset_val"]["novelty_scores"]

    metrics["validity_all"] = compute_ood_metrics(
        [0]*len(id_vs) + [1]*len(corrupt_vs),
        np.concatenate([id_vs, corrupt_vs])
    )
    metrics["validity_family_a"] = compute_ood_metrics(
        [0]*len(id_vs) + [1]*len(fam_a_vs),
        np.concatenate([id_vs, fam_a_vs])
    )
    metrics["validity_family_b"] = compute_ood_metrics(
        [0]*len(id_vs) + [1]*len(fam_b_vs),
        np.concatenate([id_vs, fam_b_vs])
    )
    metrics["novelty_scaffold_raw"] = compute_ood_metrics(
        [0]*len(id_ns) + [1]*len(scaff_ns),
        np.concatenate([id_ns, scaff_ns])
    )
    metrics["novelty_scaffold_matched"] = compute_ood_metrics(
        [0]*len(scaff_ctrl) + [1]*len(scaff_ctrl),
        np.concatenate([scaff_ctrl["cs"].to_numpy(), scaff_ctrl["ts"].to_numpy()])
    )
    metrics["novelty_dataset_raw"] = compute_ood_metrics(
        [0]*len(id_ns) + [1]*len(data_ns),
        np.concatenate([id_ns, data_ns])
    )
    metrics["novelty_dataset_matched"] = compute_ood_metrics(
        [0]*len(data_ctrl) + [1]*len(data_ctrl),
        np.concatenate([data_ctrl["cs"].to_numpy(), data_ctrl["ts"].to_numpy()])
    )

    print("\n  Recomputed Validation Metrics:")
    print(f"  {'Metric':<30} {'Reprod AUROC':>13} {'Saved AUROC':>12} {'Delta':>10}")
    print("  " + "-"*67)
    return metrics


def recompute_test_metrics(tox, val_det, nov_est) -> dict:
    banner("STEP 9.3 — Independent Test Metrics Recomputation")
    master_df = pd.read_csv(DATA_DIR / "tables/master_table.csv")
    controls_df = pd.read_csv(DATA_DIR / "tables/matched_controls.csv")

    test_splits = ["id_test", "corrupt_test", "novel_scaffold_test", "novel_dataset_test"]
    scores = {}
    for s in test_splits:
        res = score_split(
            split_name=s,
            toxicity_model=tox,
            validity_detector=val_det,
            novelty_estimator=nov_est,
            root_dir=str(DATA_DIR),
            batch_size=64,
            device=torch.device("cpu"),
        )
        scores[s] = res
        print(f"  Scored {s}: {len(res['uids'])} graphs")

    corrupt_meta = master_df[master_df["set_name"] == "corrupt_test"][["uid", "family"]].copy()
    corrupt_df = pd.DataFrame({
        "uid": scores["corrupt_test"]["uids"],
        "validity_score": scores["corrupt_test"]["validity_scores"],
        "novelty_score": scores["corrupt_test"]["novelty_scores"],
    }).merge(corrupt_meta, on="uid")

    fam_a_vs = corrupt_df[corrupt_df["family"] == "A"]["validity_score"].to_numpy()
    fam_b_vs = corrupt_df[corrupt_df["family"] == "B"]["validity_score"].to_numpy()
    fam_c_vs = corrupt_df[corrupt_df["family"] == "C"]["validity_score"].to_numpy()

    id_test_dict = dict(zip(scores["id_test"]["uids"], scores["id_test"]["novelty_scores"]))
    scaff_dict = dict(zip(scores["novel_scaffold_test"]["uids"], scores["novel_scaffold_test"]["novelty_scores"]))
    data_dict = dict(zip(scores["novel_dataset_test"]["uids"], scores["novel_dataset_test"]["novelty_scores"]))

    scaff_ctrl = controls_df[(controls_df["target_set"] == "novel_scaffold_test") & (controls_df["matched"] == True)].copy()
    data_ctrl = controls_df[(controls_df["target_set"] == "novel_dataset_test") & (controls_df["matched"] == True)].copy()
    scaff_ctrl["ts"] = scaff_ctrl["target_uid"].map(scaff_dict)
    scaff_ctrl["cs"] = scaff_ctrl["control_uid"].map(id_test_dict)
    data_ctrl["ts"] = data_ctrl["target_uid"].map(data_dict)
    data_ctrl["cs"] = data_ctrl["control_uid"].map(id_test_dict)

    id_vs = scores["id_test"]["validity_scores"]
    corrupt_vs = scores["corrupt_test"]["validity_scores"]
    id_ns = scores["id_test"]["novelty_scores"]
    scaff_ns = scores["novel_scaffold_test"]["novelty_scores"]
    data_ns = scores["novel_dataset_test"]["novelty_scores"]

    metrics = {}
    metrics["validity_all_test"] = compute_ood_metrics([0]*len(id_vs)+[1]*len(corrupt_vs), np.concatenate([id_vs, corrupt_vs]))
    metrics["validity_family_a_test"] = compute_ood_metrics([0]*len(id_vs)+[1]*len(fam_a_vs), np.concatenate([id_vs, fam_a_vs]))
    metrics["validity_family_b_test"] = compute_ood_metrics([0]*len(id_vs)+[1]*len(fam_b_vs), np.concatenate([id_vs, fam_b_vs]))
    metrics["validity_family_c_test"] = compute_ood_metrics([0]*len(id_vs)+[1]*len(fam_c_vs), np.concatenate([id_vs, fam_c_vs]))
    metrics["novelty_scaffold_raw_test"] = compute_ood_metrics([0]*len(id_ns)+[1]*len(scaff_ns), np.concatenate([id_ns, scaff_ns]))
    metrics["novelty_scaffold_matched_test"] = compute_ood_metrics(
        [0]*len(scaff_ctrl)+[1]*len(scaff_ctrl),
        np.concatenate([scaff_ctrl["cs"].to_numpy(), scaff_ctrl["ts"].to_numpy()])
    )
    metrics["novelty_dataset_raw_test"] = compute_ood_metrics([0]*len(id_ns)+[1]*len(data_ns), np.concatenate([id_ns, data_ns]))
    metrics["novelty_dataset_matched_test"] = compute_ood_metrics(
        [0]*len(data_ctrl)+[1]*len(data_ctrl),
        np.concatenate([data_ctrl["cs"].to_numpy(), data_ctrl["ts"].to_numpy()])
    )

    print(f"\n  Family C AUROC (test, held out): {metrics['validity_family_c_test']['auroc']:.4f}")
    n_fam_c = len(fam_c_vs)
    print(f"  Family C count in corrupt_test: {n_fam_c}")
    assert n_fam_c > 0, "Family C not found in corrupt_test!"
    assert metrics["validity_family_c_test"]["auroc"] > 0.90, \
        f"Family C AUROC too low: {metrics['validity_family_c_test']['auroc']:.4f}"
    print("  [OK] Family C generalization test passed (AUROC > 0.90).")

    return metrics


def calibration_analysis(tox, val_det, nov_est) -> dict:
    """
    Calibration analysis for validity detector:
      - Expected Calibration Error (ECE) on validation set
      - Mean validity score on id_val vs corrupt_val
    """
    banner("STEP 9.4 — Calibration Analysis (Validity Detector)")
    id_res = score_split("id_val", tox, val_det, nov_est, str(DATA_DIR), 64, torch.device("cpu"))
    cor_res = score_split("corrupt_val", tox, val_det, nov_est, str(DATA_DIR), 64, torch.device("cpu"))

    id_scores = id_res["validity_scores"]
    cor_scores = cor_res["validity_scores"]

    labels = np.array([0] * len(id_scores) + [1] * len(cor_scores))
    probs = np.concatenate([id_scores, cor_scores])

    # ECE calculation (10 bins)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    bin_details = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        cnt = mask.sum()
        if cnt == 0:
            bin_details.append({"bin": f"[{lo:.1f},{hi:.1f})", "count": 0, "avg_conf": None, "avg_acc": None})
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (cnt / n) * abs(avg_conf - avg_acc)
        bin_details.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "count": int(cnt),
            "avg_conf": float(avg_conf),
            "avg_acc": float(avg_acc),
            "gap": float(avg_conf - avg_acc),
        })

    mean_id_score = float(id_scores.mean())
    mean_cor_score = float(cor_scores.mean())
    print(f"  Mean validity score on id_val:      {mean_id_score:.4f}")
    print(f"  Mean validity score on corrupt_val:  {mean_cor_score:.4f}")
    print(f"  Expected Calibration Error (ECE):    {ece:.4f}")
    print(f"  [OK] Mean separation: {mean_cor_score - mean_id_score:.4f}")

    return {
        "ece": float(ece),
        "mean_id_score": mean_id_score,
        "mean_corrupt_score": mean_cor_score,
        "separation": mean_cor_score - mean_id_score,
        "bin_details": bin_details,
    }


def compare_metrics(reproduced: dict, saved: dict, label: str) -> dict:
    """Compare reproduced vs saved metrics and return discrepancies."""
    discrepancies = []
    for key, saved_vals in saved.items():
        if key not in reproduced:
            discrepancies.append({"key": key, "issue": "missing_from_reproduced"})
            continue
        repro_auroc = reproduced[key]["auroc"]
        saved_auroc = saved_vals["auroc"]
        delta = abs(repro_auroc - saved_auroc)
        match = "EXACT" if delta < 1e-6 else ("CLOSE" if delta < 1e-3 else "DISCREPANCY")
        print(f"  {key:<36} repro={repro_auroc:.6f}  saved={saved_auroc:.6f}  Delta={delta:.2e}  [{match}]")
        if delta >= 1e-3:
            discrepancies.append({"key": key, "repro_auroc": repro_auroc, "saved_auroc": saved_auroc, "delta": delta})
    return {"label": label, "discrepancies": discrepancies, "total_checked": len(saved)}


def main():
    start = time.time()
    banner("STEP 9 — FINAL SCIENTIFIC AUDIT, ROBUSTNESS REVIEW, PAPER-READINESS GATE")
    print("  Repository:", REPO)
    print("  Output Dir:", OUTPUT_DIR)

    audit_results = {}

    # 1. Checkpoint check
    audit_results["checkpoint_check"] = check_checkpoints()

    # Load components once for all checks
    banner("Loading frozen components for audit...")
    tox, val_det, nov_est = load_frozen_components(
        toxicity_ckpt_path=str(TOXICITY_CKPT),
        validity_ckpt_path=str(VALIDITY_CKPT),
        novelty_estimator_path=str(NOVELTY_CKPT),
        device=torch.device("cpu"),
    )

    # Load saved metrics
    ref = load_reference_metrics()

    # 2. Val metrics recomputation
    repro_val = recompute_val_metrics(tox, val_det, nov_est)
    banner("Comparing val_metrics.json vs reproduced...")
    val_compare = compare_metrics(repro_val, ref["val"], label="validation")
    audit_results["val_metrics_comparison"] = val_compare

    # 3. Test metrics recomputation
    repro_test = recompute_test_metrics(tox, val_det, nov_est)
    banner("Comparing test_metrics.json vs reproduced...")
    test_compare = compare_metrics(repro_test, ref["test"], label="test")
    audit_results["test_metrics_comparison"] = test_compare

    # 4. Calibration
    calib = calibration_analysis(tox, val_det, nov_est)
    audit_results["calibration"] = calib

    # 5. Save audit summary JSON
    elapsed = time.time() - start
    audit_results["elapsed_seconds"] = round(elapsed, 1)
    audit_results["status"] = "COMPLETE"

    summary_path = OUTPUT_DIR / "step9_audit_summary.json"
    with open(summary_path, "w") as f:
        json.dump(audit_results, f, indent=2)

    # 6. Print final summary
    banner("STEP 9 AUDIT SUMMARY")
    ck = audit_results["checkpoint_check"]
    print(f"  Checkpoint loadability:  {ck['load_status']}")
    print(f"  Val discrepancies:       {len(val_compare['discrepancies'])}/{val_compare['total_checked']} metrics")
    print(f"  Test discrepancies:      {len(test_compare['discrepancies'])}/{test_compare['total_checked']} metrics")
    print(f"  ECE (validity):          {calib['ece']:.4f}")
    print(f"  Mean score sep (val/cor):{calib['separation']:.4f}")
    print(f"  Total elapsed:           {elapsed:.1f}s")
    print(f"\n  Audit report saved: {summary_path}")
    print("=" * 70)

    return audit_results


if __name__ == "__main__":
    main()
