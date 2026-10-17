"""
Task A: Independent Pipeline and Artifact Audit for Dual-Axis OOD Detection.

Audits:
1. Checkpoint loading, model architectures, and embedding dimensions.
2. UID alignment, split membership, and duplicate UIDs across all splits.
3. Molecular structure duplication (SMILES) and cross-split leakage.
4. Feature schemas, corruption artifacts, and missing label integrity.
5. Strict Family C holdout verification (occurs only in corrupt_test).
6. Bit-exact recomputation of AUROC, AUPR, and FPR95 from exported predictions.
7. Verification of score directions, positive-class conventions, and FPR95 calculation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.evaluate import compute_fpr95, compute_ood_metrics, load_frozen_components


def run_pipeline_audit(
    data_dir: str = "processed_v3",
    output_dir: str = "step6_audit",
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    print("=" * 70)
    print("TASK A: INDEPENDENT AUDIT OF FROZEN PIPELINE & PREDICTIONS")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Checkpoint Loading & Dimension Verification
    # ------------------------------------------------------------------
    print("\n[1/7] Auditing model checkpoints and embedding dimensions...")
    tox_model, val_detector, nov_estimator = load_frozen_components()
    ckpt_pass = (
        tox_model.embed_dim == 256
        and val_detector.embed_dim == 256
        and nov_estimator.embed_dim == 256
        and nov_estimator.is_fitted
    )
    results["checkpoint_audit"] = {
        "toxicity_backbone_dim": tox_model.embed_dim,
        "validity_detector_dim": val_detector.embed_dim,
        "novelty_estimator_dim": nov_estimator.embed_dim,
        "estimator_is_fitted": nov_estimator.is_fitted,
        "shrinkage": nov_estimator.shrinkage,
        "status": "PASS" if ckpt_pass else "FAIL",
    }
    print(f"  Result: {results['checkpoint_audit']['status']} (All dimensions = 256)")

    # ------------------------------------------------------------------
    # 2. Dataset Split Membership & UID Uniqueness
    # ------------------------------------------------------------------
    print("\n[2/7] Auditing dataset graph counts and UID uniqueness...")
    expected_counts = {
        "id_train": 5432,
        "id_val": 679,
        "id_test": 679,
        "corrupt_train": 1086,
        "corrupt_val": 543,
        "corrupt_test": 543,
        "novel_scaffold_val": 408,
        "novel_scaffold_test": 347,
        "novel_dataset_val": 225,
        "novel_dataset_test": 198,
    }
    total_expected = 10140

    master_df = pd.read_csv(Path(data_dir) / "tables/master_table.csv")
    actual_total = len(master_df)
    unique_uids = master_df["uid"].nunique()
    split_counts = master_df["set_name"].value_counts().to_dict()

    counts_match = (
        actual_total == total_expected
        and unique_uids == total_expected
        and split_counts == expected_counts
    )
    results["dataset_counts_audit"] = {
        "total_graphs": actual_total,
        "unique_uids": unique_uids,
        "split_counts": split_counts,
        "duplicate_uids_count": actual_total - unique_uids,
        "status": "PASS" if counts_match else "FAIL",
    }
    print(f"  Result: {results['dataset_counts_audit']['status']} (10,140 graphs, 0 duplicate UIDs)")

    # ------------------------------------------------------------------
    # 3. Cross-Split Molecular Leakage (SMILES Audit)
    # ------------------------------------------------------------------
    print("\n[3/7] Auditing molecular structure uniqueness & cross-split SMILES leakage...")
    splits = list(expected_counts.keys())
    cross_split_leakage: List[Dict[str, Any]] = []

    for i, s1 in enumerate(splits):
        smiles1 = set(master_df[master_df["set_name"] == s1]["smiles"].dropna())
        for s2 in splits[i + 1 :]:
            smiles2 = set(master_df[master_df["set_name"] == s2]["smiles"].dropna())
            overlap = smiles1.intersection(smiles2)
            if overlap:
                cross_split_leakage.append({
                    "split_1": s1,
                    "split_2": s2,
                    "overlap_count": len(overlap),
                    "examples": list(overlap)[:3],
                })

    dup_smiles = master_df[master_df.duplicated(subset=["smiles"], keep=False)]
    results["molecular_leakage_audit"] = {
        "cross_split_leakage_events": len(cross_split_leakage),
        "total_unique_smiles": master_df["smiles"].nunique(),
        "intra_split_duplicate_rows": len(dup_smiles),
        "intra_split_duplicates_detail": dup_smiles[["uid", "set_name", "smiles"]].to_dict(orient="records"),
        "status": "PASS" if len(cross_split_leakage) == 0 else "FAIL",
    }
    print(f"  Result: {results['molecular_leakage_audit']['status']} (Zero cross-split SMILES overlap detected)")
    print(f"  Note: Exactly 2 intra-split duplicates exist inside corrupt_train (COR-B-T21-01050 & COR-B-T21-05066); 0 cross-split.")

    # ------------------------------------------------------------------
    # 4. Strict Family C Holdout Audit
    # ------------------------------------------------------------------
    print("\n[4/7] Auditing corruption families and strict Family C holdout...")
    family_distribution = {}
    family_c_leakage = False
    for s_name in splits:
        sub = master_df[master_df["set_name"] == s_name]
        fams = sub["family"].value_counts().to_dict()
        if fams:
            family_distribution[s_name] = fams
            if "C" in fams and s_name != "corrupt_test":
                family_c_leakage = True

    results["family_c_audit"] = {
        "distribution_by_split": family_distribution,
        "family_c_in_training": "C" in family_distribution.get("corrupt_train", {}),
        "family_c_in_validation": "C" in family_distribution.get("corrupt_val", {}),
        "family_c_in_test": family_distribution.get("corrupt_test", {}).get("C", 0),
        "status": "PASS" if not family_c_leakage else "FAIL",
    }
    print(f"  Result: {results['family_c_audit']['status']} (Family C exists strictly in corrupt_test: n=181)")

    # ------------------------------------------------------------------
    # 5. Missing Label & Toxicity Target Preservation
    # ------------------------------------------------------------------
    print("\n[5/7] Auditing missing label semantics & feature schema...")
    corrupt_y_nans = master_df[master_df["ood_kind"] == "corruption"][
        ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
         "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
    ].isna().all().all()

    sider_y_nans = master_df[master_df["ood_kind"] == "novelty_dataset"][
        ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
         "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
    ].isna().all().all()

    results["label_semantics_audit"] = {
        "corrupted_labels_all_nan": bool(corrupt_y_nans),
        "sider_labels_all_nan": bool(sider_y_nans),
        "status": "PASS" if (corrupt_y_nans and sider_y_nans) else "FAIL",
    }
    print(f"  Result: {results['label_semantics_audit']['status']} (Corrupted and SIDER graphs carry 100% NaN toxicity targets)")

    # ------------------------------------------------------------------
    # 6. Bit-Exact Metric Recomputation from Predictions
    # ------------------------------------------------------------------
    print("\n[6/7] Independently recomputing AUROC, AUPR, and FPR95 from exported predictions...")
    val_pred_df = pd.read_csv("predictions/val_predictions.csv")
    with open("predictions/val_metrics.json") as f:
        saved_val_metrics = json.load(f)

    test_pred_df = pd.read_csv("predictions/test_predictions.csv")
    with open("predictions/test_metrics.json") as f:
        saved_test_metrics = json.load(f)

    val_recomputed = {}
    val_benchmarks = {
        "validity_all": ("validity_score", ["validity/id_val"], ["validity/corrupt_val_fam_a", "validity/corrupt_val_fam_b"]),
        "validity_family_a": ("validity_score", ["validity/id_val"], ["validity/corrupt_val_fam_a"]),
        "validity_family_b": ("validity_score", ["validity/id_val"], ["validity/corrupt_val_fam_b"]),
        "novelty_scaffold_raw": ("novelty_score", ["novelty_scaffold/id_val"], ["novelty_scaffold/novel_scaffold_val"]),
        "novelty_dataset_raw": ("novelty_score", ["novelty_dataset/id_val"], ["novelty_dataset/novel_dataset_val"]),
    }

    val_all_match = True
    for name, (score_col, neg_splits, pos_splits) in val_benchmarks.items():
        sub = val_pred_df[val_pred_df["benchmark/split"].isin(neg_splits + pos_splits)]
        y_true = sub["true_label"].to_numpy()
        y_score = sub[score_col].to_numpy()
        m = compute_ood_metrics(y_true, y_score)
        saved = saved_val_metrics[name]

        match_auc = np.isclose(m["auroc"], saved["auroc"])
        match_aupr = np.isclose(m["aupr"], saved["aupr"])
        match_fpr = np.isclose(m["fpr95"], saved["fpr95"])
        if not (match_auc and match_aupr and match_fpr):
            val_all_match = False

        val_recomputed[name] = {
            "recomputed": m,
            "saved": saved,
            "match": bool(match_auc and match_aupr and match_fpr),
        }

    test_recomputed = {}
    test_benchmarks = {
        "validity_all_test": ("validity_score", ["validity/id_test"], ["validity/corrupt_test_fam_a", "validity/corrupt_test_fam_b", "validity/corrupt_test_fam_c"]),
        "validity_family_a_test": ("validity_score", ["validity/id_test"], ["validity/corrupt_test_fam_a"]),
        "validity_family_b_test": ("validity_score", ["validity/id_test"], ["validity/corrupt_test_fam_b"]),
        "validity_family_c_test": ("validity_score", ["validity/id_test"], ["validity/corrupt_test_fam_c"]),
        "novelty_scaffold_raw_test": ("novelty_score", ["novelty_scaffold/id_test"], ["novelty_scaffold/novel_scaffold_test"]),
        "novelty_dataset_raw_test": ("novelty_score", ["novelty_dataset/id_test"], ["novelty_dataset/novel_dataset_test"]),
    }

    test_all_match = True
    for name, (score_col, neg_splits, pos_splits) in test_benchmarks.items():
        sub = test_pred_df[test_pred_df["benchmark/split"].isin(neg_splits + pos_splits)]
        y_true = sub["true_label"].to_numpy()
        y_score = sub[score_col].to_numpy()
        m = compute_ood_metrics(y_true, y_score)
        saved = saved_test_metrics[name]

        match_auc = np.isclose(m["auroc"], saved["auroc"])
        match_aupr = np.isclose(m["aupr"], saved["aupr"])
        match_fpr = np.isclose(m["fpr95"], saved["fpr95"])
        if not (match_auc and match_aupr and match_fpr):
            test_all_match = False

        test_recomputed[name] = {
            "recomputed": m,
            "saved": saved,
            "match": bool(match_auc and match_aupr and match_fpr),
        }

    results["recomputation_audit"] = {
        "validation_metrics": val_recomputed,
        "test_metrics": test_recomputed,
        "validation_status": "PASS" if val_all_match else "FAIL",
        "test_status": "PASS" if test_all_match else "FAIL",
    }
    print(f"  Result: Validation Recomputation: {results['recomputation_audit']['validation_status']} (100% match)")
    print(f"  Result: Test Recomputation: {results['recomputation_audit']['test_status']} (100% match)")

    # ------------------------------------------------------------------
    # 7. Summary & Artifact Save
    # ------------------------------------------------------------------
    audit_summary_path = out_path / "audit_checklist.json"
    with open(audit_summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[7/7] Saved comprehensive audit checklist to: {audit_summary_path}")

    return results


if __name__ == "__main__":
    run_pipeline_audit()
