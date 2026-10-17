"""
Step 7: Evaluation script comparing ECFP4/Tanimoto novelty against Step 5 & Step 6 baselines.

Evaluates on Validation Splits:
- id_val vs novel_scaffold_val (Raw)
- id_val controls vs novel_scaffold_val (Matched Controls)
- id_val vs novel_dataset_val (Raw)
- id_val controls vs novel_dataset_val (Matched Controls)

Outputs:
- step7_novelty/val_ecfp4_predictions.csv
- step7_novelty/val_ecfp4_metrics.json
- step7_novelty/method_comparison_table.csv
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from step7_novelty.ecfp4_estimator import ECFP4TanimotoNoveltyEstimator


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[idx[0]]) if len(idx) > 0 else 1.0

    auc = float(roc_auc_score(y_true, y_score))
    aupr = float(average_precision_score(y_true, y_score))

    # Bootstrap 95% CI for AUROC
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        b_idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[b_idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y_true[b_idx], y_score[b_idx])))

    ci_low = float(np.percentile(aucs, 2.5)) if len(aucs) > 0 else float("nan")
    ci_high = float(np.percentile(aucs, 97.5)) if len(aucs) > 0 else float("nan")

    return {
        "auroc": auc,
        "auroc_ci_95": [ci_low, ci_high],
        "aupr": aupr,
        "fpr95": fpr95,
        "n_neg": int(np.sum(y_true == 0)),
        "n_pos": int(np.sum(y_true == 1)),
    }


def run_evaluation(
    data_dir: str = "processed_v3",
    output_dir: str = "step7_novelty",
    seed: int = 42,
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    master_path = Path(data_dir) / "tables/master_table.csv"
    controls_path = Path(data_dir) / "tables/matched_controls.csv"

    print("=" * 70)
    print("STEP 7: ECFP4 / MAXIMUM TANIMOTO MOLECULAR NOVELTY EVALUATION")
    print("=" * 70)

    # 1. Load data
    print("\n[1/5] Loading master table and matched controls...")
    master_df = pd.read_csv(master_path).set_index("uid")
    controls_df = pd.read_csv(controls_path)

    # 2. Fit ECFP4 Estimator
    print("\n[2/5] Fitting ECFP4 Tanimoto Estimator on id_train...")
    train_sub = master_df[master_df["set_name"] == "id_train"]
    estimator = ECFP4TanimotoNoveltyEstimator(radius=2, n_bits=2048)
    estimator.fit(
        smiles_list=train_sub["smiles"].tolist(),
        uids=train_sub.index.tolist(),
        fit_split="id_train",
    )
    print(f"  Fitted reference database with {estimator.n_ref} molecules.")

    # 3. Score validation splits
    print("\n[3/5] Scoring validation splits (id_val, novel_scaffold_val, novel_dataset_val)...")
    val_splits = ["id_val", "novel_scaffold_val", "novel_dataset_val"]
    scores: Dict[str, np.ndarray] = {}
    uids: Dict[str, List[str]] = {}

    for s in val_splits:
        sub = master_df[master_df["set_name"] == s]
        uids[s] = sub.index.tolist()
        scores[s] = estimator.score(sub["smiles"].tolist())
        print(
            f"  - Scored {s:<20}: {len(scores[s]):>4} graphs | "
            f"Mean Novelty: {scores[s].mean():.4f} | "
            f"Std: {scores[s].std():.4f} | "
            f"Range: [{scores[s].min():.4f}, {scores[s].max():.4f}]"
        )

    # 4. Compute Benchmark Metrics
    print("\n[4/5] Computing Benchmark Performance Metrics...")
    metrics: Dict[str, Any] = {}

    # Scaffold Raw
    y_sr = np.array([0] * len(scores["id_val"]) + [1] * len(scores["novel_scaffold_val"]))
    s_sr = np.concatenate([scores["id_val"], scores["novel_scaffold_val"]])
    metrics["novelty_scaffold_raw"] = compute_metrics_with_ci(y_sr, s_sr, seed=seed)

    # Scaffold Matched
    scaff_map = dict(zip(uids["novel_scaffold_val"], scores["novel_scaffold_val"]))
    val_map = dict(zip(uids["id_val"], scores["id_val"]))
    scaff_sub = controls_df[(controls_df["target_set"] == "novel_scaffold_val") & (controls_df["matched"] == True)]
    s_sm = np.concatenate([
        scaff_sub["control_uid"].map(val_map).to_numpy(),
        scaff_sub["target_uid"].map(scaff_map).to_numpy(),
    ])
    y_sm = np.array([0] * len(scaff_sub) + [1] * len(scaff_sub))
    metrics["novelty_scaffold_matched"] = compute_metrics_with_ci(y_sm, s_sm, seed=seed)

    # Dataset Raw
    y_dr = np.array([0] * len(scores["id_val"]) + [1] * len(scores["novel_dataset_val"]))
    s_dr = np.concatenate([scores["id_val"], scores["novel_dataset_val"]])
    metrics["novelty_dataset_raw"] = compute_metrics_with_ci(y_dr, s_dr, seed=seed)

    # Dataset Matched
    data_map = dict(zip(uids["novel_dataset_val"], scores["novel_dataset_val"]))
    data_sub = controls_df[(controls_df["target_set"] == "novel_dataset_val") & (controls_df["matched"] == True)]
    s_dm = np.concatenate([
        data_sub["control_uid"].map(val_map).to_numpy(),
        data_sub["target_uid"].map(data_map).to_numpy(),
    ])
    y_dm = np.array([0] * len(data_sub) + [1] * len(data_sub))
    metrics["novelty_dataset_matched"] = compute_metrics_with_ci(y_dm, s_dm, seed=seed)

    # Print Summary Table
    print("\n" + "-" * 90)
    print(f"{'Benchmark / Evaluation Setting':<32} | {'AUROC [95% CI]':<22} | {'AUPR':>7} | {'FPR95':>7} | {'N (Neg/Pos)':>11}")
    print("-" * 90)
    for k, m in metrics.items():
        ci_str = f"{m['auroc']:.4f} [{m['auroc_ci_95'][0]:.3f}-{m['auroc_ci_95'][1]:.3f}]"
        n_str = f"{m['n_neg']}/{m['n_pos']}"
        print(f"{k:<32} | {ci_str:<22} | {m['aupr']:>7.4f} | {m['fpr95']:>7.4f} | {n_str:>11}")
    print("-" * 90)

    # 5. Export Deliverables
    print("\n[5/5] Exporting predictions, metrics, and comparisons...")

    # A. Export predictions CSV
    pred_rows = []
    # Scaffold benchmark rows
    for u, s in zip(uids["id_val"], scores["id_val"]):
        pred_rows.append({"uid": u, "benchmark/split": "novelty_scaffold/id_val", "true_label": 0, "novelty_score": s})
    for u, s in zip(uids["novel_scaffold_val"], scores["novel_scaffold_val"]):
        pred_rows.append({"uid": u, "benchmark/split": "novelty_scaffold/novel_scaffold_val", "true_label": 1, "novelty_score": s})
    # Dataset benchmark rows
    for u, s in zip(uids["id_val"], scores["id_val"]):
        pred_rows.append({"uid": u, "benchmark/split": "novelty_dataset/id_val", "true_label": 0, "novelty_score": s})
    for u, s in zip(uids["novel_dataset_val"], scores["novel_dataset_val"]):
        pred_rows.append({"uid": u, "benchmark/split": "novelty_dataset/novel_dataset_val", "true_label": 1, "novelty_score": s})

    pred_df = pd.DataFrame(pred_rows)
    pred_csv_path = out_path / "val_ecfp4_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"  - Saved predictions: {pred_csv_path} ({len(pred_df)} rows)")

    # B. Export metrics JSON
    metrics_json_path = out_path / "val_ecfp4_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  - Saved metrics: {metrics_json_path}")

    # C. Comparison with Step 5 / Step 6 Baselines
    # Load Step 5 metrics
    with open("predictions/val_metrics.json") as f:
        step5_val = json.load(f)

    # Load Step 6 results
    step6_results_df = pd.read_csv("step6_audit/novelty_experiment_results.csv")
    energy_row = step6_results_df[step6_results_df["method"].str.contains("Free Energy")].iloc[0]
    knn_row = step6_results_df[step6_results_df["method"].str.contains("k=20")].iloc[0]

    comparison_rows = [
        {
            "Method / Model": "GNN Mahalanobis (Step 5 Frozen Baseline)",
            "Scaff Raw AUROC": f"{step5_val['novelty_scaffold_raw']['auroc']:.4f}",
            "Scaff Matched AUROC": f"{step5_val['novelty_scaffold_matched']['auroc']:.4f}",
            "Data Raw AUROC": f"{step5_val['novelty_dataset_raw']['auroc']:.4f}",
            "Data Matched AUROC": f"{step5_val['novelty_dataset_matched']['auroc']:.4f}",
        },
        {
            "Method / Model": "Multi-Task Free Energy Score (Step 6)",
            "Scaff Raw AUROC": f"{energy_row['scaff_raw_auroc']:.4f}",
            "Scaff Matched AUROC": f"{energy_row['scaff_matched_auroc']:.4f}",
            "Data Raw AUROC": f"{energy_row['dataset_raw_auroc']:.4f}",
            "Data Matched AUROC": f"{energy_row['dataset_matched_auroc']:.4f}",
        },
        {
            "Method / Model": "Latent Cosine kNN (k=20) (Step 6)",
            "Scaff Raw AUROC": f"{knn_row['scaff_raw_auroc']:.4f}",
            "Scaff Matched AUROC": f"{knn_row['scaff_matched_auroc']:.4f}",
            "Data Raw AUROC": f"{knn_row['dataset_raw_auroc']:.4f}",
            "Data Matched AUROC": f"{knn_row['dataset_matched_auroc']:.4f}",
        },
        {
            "Method / Model": "ECFP4 / Max Tanimoto Distance (Step 7)",
            "Scaff Raw AUROC": f"{metrics['novelty_scaffold_raw']['auroc']:.4f}",
            "Scaff Matched AUROC": f"{metrics['novelty_scaffold_matched']['auroc']:.4f}",
            "Data Raw AUROC": f"{metrics['novelty_dataset_raw']['auroc']:.4f}",
            "Data Matched AUROC": f"{metrics['novelty_dataset_matched']['auroc']:.4f}",
        },
    ]

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = out_path / "method_comparison_table.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"  - Saved comparison table: {comp_csv_path}")

    print("\n" + "=" * 90)
    print("STEP 7 METHOD COMPARISON TABLE")
    print("=" * 90)
    print(comp_df.to_string(index=False))
    print("=" * 90)

    return metrics


if __name__ == "__main__":
    run_evaluation()
