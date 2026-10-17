"""
Task B: Graph Statistics Baselines for Validity and Novelty Detection.

Evaluates:
1. Size-only baselines: Atom count (N), Bond count (E), Graph size (N + E), Heavy atom count.
2. Degree statistics and element count baselines.
3. Tabular classifiers trained on counting features: Logistic Regression & Random Forest.
4. Comparison against GINE Validity Detector and Mahalanobis Novelty Detector.
5. Evaluates both raw and size-matched control settings and reports matching coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.evaluate import compute_fpr95, compute_ood_metrics


def run_baseline_experiments(
    data_dir: str = "processed_v3",
    output_dir: str = "step6_audit",
    seed: int = 42,
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    print("=" * 70)
    print("TASK B: EVALUATING GRAPH-STATISTICS BASELINES VS GNN MODELS")
    print("=" * 70)

    # 1. Load graph tensors and metadata
    sets = [
        "id_train", "id_val",
        "corrupt_train", "corrupt_val",
        "novel_scaffold_val", "novel_dataset_val"
    ]
    print("\n[1/4] Loading graph tensors from disk...")
    graphs: Dict[str, List[Dict[str, Any]]] = {}
    for s in sets:
        graphs[s] = torch.load(Path(data_dir) / f"graphs/{s}.pt", weights_only=True)
        print(f"  - Loaded {s:<20}: {len(graphs[s]):>5} graphs")

    master_df = pd.read_csv(Path(data_dir) / "tables/master_table.csv").set_index("uid")
    controls_df = pd.read_csv(Path(data_dir) / "tables/matched_controls.csv")

    # Feature extractor: 32 summary statistics
    # [n_nodes, n_edges] + 30-dim atom feature sums (13 elements, 6 degrees, 4 charges, 5 Hs, 2 ring/aromatic)
    def extract_count_features(g: Dict[str, Any]) -> np.ndarray:
        n_nodes = float(g["x"].shape[0])
        n_edges = float(g["edge_index"].shape[1] // 2)
        x_sum = g["x"].sum(dim=0).numpy()
        return np.concatenate([[n_nodes, n_edges], x_sum]).astype(np.float32)

    X_counts = {s: np.array([extract_count_features(g) for g in graphs[s]]) for s in sets}
    uids = {s: [g["uid"] for g in graphs[s]] for s in sets}

    # ------------------------------------------------------------------
    # 2. Validity Axis Baselines (id_val vs corrupt_val)
    # ------------------------------------------------------------------
    print("\n[2/4] Evaluating Validity Baselines on id_val vs corrupt_val...")
    corrupt_val_fams = [master_df.loc[u, "family"] for u in uids["corrupt_val"]]
    fam_a_mask = np.array([f == "A" for f in corrupt_val_fams])
    fam_b_mask = np.array([f == "B" for f in corrupt_val_fams])

    n_id_val = len(X_counts["id_val"])
    y_val_all = [0] * n_id_val + [1] * len(X_counts["corrupt_val"])
    y_val_a = [0] * n_id_val + [1] * int(fam_a_mask.sum())
    y_val_b = [0] * n_id_val + [1] * int(fam_b_mask.sum())

    # Supervised count models trained on id_train + corrupt_train
    X_train_val = np.concatenate([X_counts["id_train"], X_counts["corrupt_train"]])
    y_train_val = np.array([0] * len(X_counts["id_train"]) + [1] * len(X_counts["corrupt_train"]))
    X_val_val = np.concatenate([X_counts["id_val"], X_counts["corrupt_val"]])

    lr_val = LogisticRegression(max_iter=1000, random_state=seed).fit(X_train_val, y_train_val)
    rf_val = RandomForestClassifier(n_estimators=100, random_state=seed).fit(X_train_val, y_train_val)

    prob_lr_val = lr_val.predict_proba(X_val_val)[:, 1]
    prob_rf_val = rf_val.predict_proba(X_val_val)[:, 1]

    # Load frozen GNN validity detector scores for exact comparison
    val_pred_df = pd.read_csv("predictions/val_predictions.csv")
    gnn_val_map = dict(zip(val_pred_df["uid"], val_pred_df["validity_score"]))
    s_gnn_val_id = [gnn_val_map[u] for u in uids["id_val"]]
    s_gnn_val_corrupt = [gnn_val_map[u] for u in uids["corrupt_val"]]
    s_gnn_all = s_gnn_val_id + s_gnn_val_corrupt

    validity_baselines = {
        "Atom Count (N)": np.concatenate([X_counts["id_val"][:, 0], X_counts["corrupt_val"][:, 0]]),
        "Bond Count (E)": np.concatenate([X_counts["id_val"][:, 1], X_counts["corrupt_val"][:, 1]]),
        "Graph Size (N + E)": np.concatenate([
            X_counts["id_val"][:, 0] + X_counts["id_val"][:, 1],
            X_counts["corrupt_val"][:, 0] + X_counts["corrupt_val"][:, 1],
        ]),
        "Logistic Regression (32 Counts)": prob_lr_val,
        "Random Forest (32 Counts)": prob_rf_val,
        "GINE Validity Detector (Ours)": np.array(s_gnn_all),
    }

    val_table_rows = []
    for model_name, s_all in validity_baselines.items():
        m_all = compute_ood_metrics(y_val_all, s_all)

        s_a = np.concatenate([s_all[:n_id_val], s_all[n_id_val:][fam_a_mask]])
        m_a = compute_ood_metrics(y_val_a, s_a)

        s_b = np.concatenate([s_all[:n_id_val], s_all[n_id_val:][fam_b_mask]])
        m_b = compute_ood_metrics(y_val_b, s_b)

        val_table_rows.append({
            "model": model_name,
            "all_auroc": m_all["auroc"],
            "all_fpr95": m_all["fpr95"],
            "fam_a_auroc": m_a["auroc"],
            "fam_a_fpr95": m_a["fpr95"],
            "fam_b_auroc": m_b["auroc"],
            "fam_b_fpr95": m_b["fpr95"],
        })

    df_val_results = pd.DataFrame(val_table_rows)
    print("\n--- VALIDITY BASELINE COMPARISON ---")
    print(df_val_results.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Novelty Axis Baselines (Raw vs Matched Controls)
    # ------------------------------------------------------------------
    print("\n[3/4] Evaluating Novelty Baselines on Scaffold and Dataset Novelty...")

    # Matching coverage
    scaff_matched_df = controls_df[(controls_df["target_set"] == "novel_scaffold_val") & (controls_df["matched"] == True)]
    scaff_coverage = len(scaff_matched_df) / len(uids["novel_scaffold_val"]) * 100.0

    data_matched_df = controls_df[(controls_df["target_set"] == "novel_dataset_val") & (controls_df["matched"] == True)]
    data_coverage = len(data_matched_df) / len(uids["novel_dataset_val"]) * 100.0

    print(f"  - Scaffold Novelty Matching Coverage: {len(scaff_matched_df)}/{len(uids['novel_scaffold_val'])} ({scaff_coverage:.1f}%)")
    print(f"  - Dataset Novelty Matching Coverage:  {len(data_matched_df)}/{len(uids['novel_dataset_val'])} ({data_coverage:.1f}%)")

    # Load frozen GNN Mahalanobis novelty scores
    df_val_scores = pd.read_csv("predictions/val_scores_by_split.csv")
    gnn_nov_map = dict(zip(df_val_scores["uid"], df_val_scores["novelty_score"]))

    # Unsupervised Mahalanobis on count features (fitted strictly on id_train)
    from sklearn.covariance import LedoitWolf
    from scipy.linalg import pinvh

    lw_counts = LedoitWolf().fit(X_counts["id_train"])
    mu_counts = X_counts["id_train"].mean(axis=0)
    prec_counts = pinvh(lw_counts.covariance_ + 1e-6 * np.eye(X_counts["id_train"].shape[1]))

    def maha_counts_score(X: np.ndarray) -> np.ndarray:
        diff = X - mu_counts
        return np.sqrt(np.clip(np.sum((diff @ prec_counts) * diff, axis=1), 0, None))

    count_models = {
        "Atom Count (N)": {s: X_counts[s][:, 0] for s in sets},
        "Bond Count (E)": {s: X_counts[s][:, 1] for s in sets},
        "Graph Size (N + E)": {s: X_counts[s][:, 0] + X_counts[s][:, 1] for s in sets},
        "Mahalanobis on Counts (32d)": {s: maha_counts_score(X_counts[s]) for s in sets},
        "GNN Mahalanobis (Step 5 frozen)": {
            s: np.array([gnn_nov_map[u] for u in uids[s]]) for s in ["id_val", "novel_scaffold_val", "novel_dataset_val"]
        },
    }

    nov_table_rows = []
    for model_name, score_dict in count_models.items():
        s_id = score_dict["id_val"]
        s_scaff = score_dict["novel_scaffold_val"]
        s_data = score_dict["novel_dataset_val"]

        # Scaffold Raw: id_val (0) vs novel_scaffold_val (1)
        y_scaff_raw = [0] * len(s_id) + [1] * len(s_scaff)
        m_scaff_raw = compute_ood_metrics(y_scaff_raw, np.concatenate([s_id, s_scaff]))

        # Scaffold Matched
        scaff_map = dict(zip(uids["novel_scaffold_val"], s_scaff))
        id_map = dict(zip(uids["id_val"], s_id))
        s_scaff_match = np.concatenate([
            scaff_matched_df["control_uid"].map(id_map).to_numpy(),
            scaff_matched_df["target_uid"].map(scaff_map).to_numpy(),
        ])
        y_scaff_match = [0] * len(scaff_matched_df) + [1] * len(scaff_matched_df)
        m_scaff_match = compute_ood_metrics(y_scaff_match, s_scaff_match)

        # Dataset Raw: id_val (0) vs novel_dataset_val (1)
        y_data_raw = [0] * len(s_id) + [1] * len(s_data)
        m_data_raw = compute_ood_metrics(y_data_raw, np.concatenate([s_id, s_data]))

        # Dataset Matched
        data_map = dict(zip(uids["novel_dataset_val"], s_data))
        s_data_match = np.concatenate([
            data_matched_df["control_uid"].map(id_map).to_numpy(),
            data_matched_df["target_uid"].map(data_map).to_numpy(),
        ])
        y_data_match = [0] * len(data_matched_df) + [1] * len(data_matched_df)
        m_data_match = compute_ood_metrics(y_data_match, s_data_match)

        nov_table_rows.append({
            "model": model_name,
            "scaff_raw_auroc": m_scaff_raw["auroc"],
            "scaff_raw_fpr95": m_scaff_raw["fpr95"],
            "scaff_matched_auroc": m_scaff_match["auroc"],
            "scaff_matched_fpr95": m_scaff_match["fpr95"],
            "dataset_raw_auroc": m_data_raw["auroc"],
            "dataset_raw_fpr95": m_data_raw["fpr95"],
            "dataset_matched_auroc": m_data_match["auroc"],
            "dataset_matched_fpr95": m_data_match["fpr95"],
        })

    df_nov_results = pd.DataFrame(nov_table_rows)
    print("\n--- NOVELTY BASELINE COMPARISON ---")
    print(df_nov_results.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Save Deliverables
    # ------------------------------------------------------------------
    results["validity_baselines"] = val_table_rows
    results["novelty_baselines"] = nov_table_rows
    results["matching_coverage"] = {
        "novel_scaffold_val": {
            "total": len(uids["novel_scaffold_val"]),
            "matched": len(scaff_matched_df),
            "coverage_percent": scaff_coverage,
        },
        "novel_dataset_val": {
            "total": len(uids["novel_dataset_val"]),
            "matched": len(data_matched_df),
            "coverage_percent": data_coverage,
        },
    }

    out_json = out_path / "baseline_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    df_val_results.to_csv(out_path / "validity_baselines.csv", index=False)
    df_nov_results.to_csv(out_path / "novelty_baselines.csv", index=False)

    print(f"\n[4/4] Saved baseline experiment results to {out_json}")
    return results


if __name__ == "__main__":
    run_baseline_experiments()
