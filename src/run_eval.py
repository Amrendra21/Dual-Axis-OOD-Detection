"""
Validation Evaluation Runner for Dual-Axis OOD Detection.

Executes Step 5 Validation Evaluation:
1. Loads all three frozen components (toxicity GNN, validity detector, novelty estimator).
2. Verifies shapes, finite score checks, and score directions.
3. Evaluates validation splits:
   - Validity axis: id_val (679) vs corrupt_val (543: Fam A = 272, Fam B = 271)
   - Novelty axis: id_val (679) vs novel_scaffold_val (408) & novel_dataset_val (225)
   - Novelty matched controls: evaluated against matched controls from matched_controls.csv
4. Exports standardized prediction CSVs and metrics logs to predictions/:
   - predictions/val_predictions.csv
   - predictions/val_scores_by_split.csv
   - predictions/val_toxicity_predictions.csv
   - predictions/val_metrics.json
   - predictions/evaluation_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.evaluate import (
    build_benchmark_predictions_df,
    compute_ood_metrics,
    load_frozen_components,
    score_split,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_eval")


def run_validation_evaluation(
    data_dir: str = "processed_v3",
    output_dir: str = "predictions",
    toxicity_checkpoint: str = "checkpoints/toxicity_gine/best_model.pt",
    validity_checkpoint: str = "checkpoints/validity_gine/best_validity_model.pt",
    novelty_estimator_path: str = "checkpoints/novelty_mahal/estimator.npz",
    batch_size: int = 64,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run full validation evaluation and export predictions.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    print("=" * 70)
    print("STEP 5: INTEGRATED MODEL EVALUATION -- VALIDATION SUITE")
    print("=" * 70)

    # 1. Load frozen components
    print("\n[1/5] Loading frozen components...")
    tox_model, val_detector, nov_estimator = load_frozen_components(
        toxicity_ckpt_path=toxicity_checkpoint,
        validity_ckpt_path=validity_checkpoint,
        novelty_estimator_path=novelty_estimator_path,
        device=dev,
    )
    print(f"  - Toxicity Backbone embed_dim: {tox_model.embed_dim}")
    print(f"  - Validity Detector embed_dim: {val_detector.embed_dim}")
    print(f"  - Novelty Estimator embed_dim: {nov_estimator.embed_dim}")
    print("  All components loaded and dimensions verified successfully.")

    # 2. Score validation splits
    print("\n[2/5] Scoring validation splits...")
    splits_to_eval = [
        "id_val",
        "corrupt_val",
        "novel_scaffold_val",
        "novel_dataset_val",
    ]
    scores_by_split: Dict[str, Dict[str, Any]] = {}
    for s_name in splits_to_eval:
        res = score_split(
            split_name=s_name,
            toxicity_model=tox_model,
            validity_detector=val_detector,
            novelty_estimator=nov_estimator,
            root_dir=data_dir,
            batch_size=batch_size,
            device=dev,
        )
        scores_by_split[s_name] = res
        print(
            f"  - Scored {s_name:<20}: {len(res['uids']):>4} graphs | "
            f"Mean Val Score: {res['validity_scores'].mean():.4f} | "
            f"Mean Nov Score: {res['novelty_scores'].mean():.4f}"
        )

    # 3. Load metadata tables
    print("\n[3/5] Loading metadata for per-family and matched-control evaluation...")
    master_df = pd.read_csv(Path(data_dir) / "tables/master_table.csv")
    controls_df = pd.read_csv(Path(data_dir) / "tables/matched_controls.csv")

    corrupt_val_meta = master_df[master_df["set_name"] == "corrupt_val"][["uid", "family"]].copy()
    corrupt_scores_df = pd.DataFrame({
        "uid": scores_by_split["corrupt_val"]["uids"],
        "validity_score": scores_by_split["corrupt_val"]["validity_scores"],
        "novelty_score": scores_by_split["corrupt_val"]["novelty_scores"],
    }).merge(corrupt_val_meta, on="uid")

    fam_a_uids = corrupt_scores_df[corrupt_scores_df["family"] == "A"]["uid"].tolist()
    fam_a_val_scores = corrupt_scores_df[corrupt_scores_df["family"] == "A"]["validity_score"].to_numpy()
    fam_a_nov_scores = corrupt_scores_df[corrupt_scores_df["family"] == "A"]["novelty_score"].to_numpy()

    fam_b_uids = corrupt_scores_df[corrupt_scores_df["family"] == "B"]["uid"].tolist()
    fam_b_val_scores = corrupt_scores_df[corrupt_scores_df["family"] == "B"]["validity_score"].to_numpy()
    fam_b_nov_scores = corrupt_scores_df[corrupt_scores_df["family"] == "B"]["novelty_score"].to_numpy()

    # Fast UID-to-score mappings
    id_val_dict = dict(zip(scores_by_split["id_val"]["uids"], scores_by_split["id_val"]["novelty_scores"]))
    scaff_val_dict = dict(zip(scores_by_split["novel_scaffold_val"]["uids"], scores_by_split["novel_scaffold_val"]["novelty_scores"]))
    data_val_dict = dict(zip(scores_by_split["novel_dataset_val"]["uids"], scores_by_split["novel_dataset_val"]["novelty_scores"]))

    # Matched controls extraction
    scaff_matched = controls_df[(controls_df["target_set"] == "novel_scaffold_val") & (controls_df["matched"] == True)].copy()
    scaff_matched["target_score"] = scaff_matched["target_uid"].map(scaff_val_dict)
    scaff_matched["control_score"] = scaff_matched["control_uid"].map(id_val_dict)

    data_matched = controls_df[(controls_df["target_set"] == "novel_dataset_val") & (controls_df["matched"] == True)].copy()
    data_matched["target_score"] = data_matched["target_uid"].map(data_val_dict)
    data_matched["control_score"] = data_matched["control_uid"].map(id_val_dict)

    # 4. Compute Metrics
    print("\n[4/5] Computing Benchmark Performance Metrics...")
    metrics: Dict[str, Any] = {}

    # --- Validity Metrics ---
    # Validity All: id_val (0) vs corrupt_val (1)
    y_val_all = [0] * len(scores_by_split["id_val"]["uids"]) + [1] * len(scores_by_split["corrupt_val"]["uids"])
    s_val_all = np.concatenate([scores_by_split["id_val"]["validity_scores"], scores_by_split["corrupt_val"]["validity_scores"]])
    metrics["validity_all"] = compute_ood_metrics(y_val_all, s_val_all)

    # Validity Family A: id_val (0) vs corrupt_val Fam A (1)
    y_val_a = [0] * len(scores_by_split["id_val"]["uids"]) + [1] * len(fam_a_uids)
    s_val_a = np.concatenate([scores_by_split["id_val"]["validity_scores"], fam_a_val_scores])
    metrics["validity_family_a"] = compute_ood_metrics(y_val_a, s_val_a)

    # Validity Family B: id_val (0) vs corrupt_val Fam B (1)
    y_val_b = [0] * len(scores_by_split["id_val"]["uids"]) + [1] * len(fam_b_uids)
    s_val_b = np.concatenate([scores_by_split["id_val"]["validity_scores"], fam_b_val_scores])
    metrics["validity_family_b"] = compute_ood_metrics(y_val_b, s_val_b)

    # --- Novelty Metrics ---
    # Novel Scaffold (Raw): id_val (0) vs novel_scaffold_val (1)
    y_scaff_raw = [0] * len(scores_by_split["id_val"]["uids"]) + [1] * len(scores_by_split["novel_scaffold_val"]["uids"])
    s_scaff_raw = np.concatenate([scores_by_split["id_val"]["novelty_scores"], scores_by_split["novel_scaffold_val"]["novelty_scores"]])
    metrics["novelty_scaffold_raw"] = compute_ood_metrics(y_scaff_raw, s_scaff_raw)

    # Novel Scaffold (Matched Controls)
    y_scaff_match = [0] * len(scaff_matched) + [1] * len(scaff_matched)
    s_scaff_match = np.concatenate([scaff_matched["control_score"].to_numpy(), scaff_matched["target_score"].to_numpy()])
    metrics["novelty_scaffold_matched"] = compute_ood_metrics(y_scaff_match, s_scaff_match)

    # Novel Dataset (Raw): id_val (0) vs novel_dataset_val (1)
    y_data_raw = [0] * len(scores_by_split["id_val"]["uids"]) + [1] * len(scores_by_split["novel_dataset_val"]["uids"])
    s_data_raw = np.concatenate([scores_by_split["id_val"]["novelty_scores"], scores_by_split["novel_dataset_val"]["novelty_scores"]])
    metrics["novelty_dataset_raw"] = compute_ood_metrics(y_data_raw, s_data_raw)

    # Novel Dataset (Matched Controls)
    y_data_match = [0] * len(data_matched) + [1] * len(data_matched)
    s_data_match = np.concatenate([data_matched["control_score"].to_numpy(), data_matched["target_score"].to_numpy()])
    metrics["novelty_dataset_matched"] = compute_ood_metrics(y_data_match, s_data_match)

    # Print Metrics Table
    print("\n" + "-" * 75)
    print(f"{'Benchmark / Evaluation Setting':<36} | {'AUROC':>7} | {'AUPR':>7} | {'FPR95':>7} | {'N (Neg/Pos)':>11}")
    print("-" * 75)
    for k, m in metrics.items():
        n_str = f"{m['n_neg']}/{m['n_pos']}"
        print(f"{k:<36} | {m['auroc']:>7.4f} | {m['aupr']:>7.4f} | {m['fpr95']:>7.4f} | {n_str:>11}")
    print("-" * 75)

    # 5. Export Prediction Files
    print("\n[5/5] Exporting standardized prediction files...")

    # A. val_predictions.csv (with agreed schema: uid, benchmark/split, true_label, validity_score, novelty_score)
    pred_dfs: List[pd.DataFrame] = []

    # Validity benchmark rows
    df_val_bm = pd.concat([
        pd.DataFrame({
            "uid": scores_by_split["id_val"]["uids"],
            "benchmark/split": "validity/id_val",
            "true_label": 0,
            "validity_score": scores_by_split["id_val"]["validity_scores"],
            "novelty_score": scores_by_split["id_val"]["novelty_scores"],
        }),
        pd.DataFrame({
            "uid": fam_a_uids,
            "benchmark/split": "validity/corrupt_val_fam_a",
            "true_label": 1,
            "validity_score": fam_a_val_scores,
            "novelty_score": fam_a_nov_scores,
        }),
        pd.DataFrame({
            "uid": fam_b_uids,
            "benchmark/split": "validity/corrupt_val_fam_b",
            "true_label": 1,
            "validity_score": fam_b_val_scores,
            "novelty_score": fam_b_nov_scores,
        }),
    ], ignore_index=True)
    pred_dfs.append(df_val_bm)

    # Novelty scaffold benchmark rows
    df_scaff_bm = pd.concat([
        pd.DataFrame({
            "uid": scores_by_split["id_val"]["uids"],
            "benchmark/split": "novelty_scaffold/id_val",
            "true_label": 0,
            "validity_score": scores_by_split["id_val"]["validity_scores"],
            "novelty_score": scores_by_split["id_val"]["novelty_scores"],
        }),
        pd.DataFrame({
            "uid": scores_by_split["novel_scaffold_val"]["uids"],
            "benchmark/split": "novelty_scaffold/novel_scaffold_val",
            "true_label": 1,
            "validity_score": scores_by_split["novel_scaffold_val"]["validity_scores"],
            "novelty_score": scores_by_split["novel_scaffold_val"]["novelty_scores"],
        }),
    ], ignore_index=True)
    pred_dfs.append(df_scaff_bm)

    # Novelty dataset benchmark rows
    df_data_bm = pd.concat([
        pd.DataFrame({
            "uid": scores_by_split["id_val"]["uids"],
            "benchmark/split": "novelty_dataset/id_val",
            "true_label": 0,
            "validity_score": scores_by_split["id_val"]["validity_scores"],
            "novelty_score": scores_by_split["id_val"]["novelty_scores"],
        }),
        pd.DataFrame({
            "uid": scores_by_split["novel_dataset_val"]["uids"],
            "benchmark/split": "novelty_dataset/novel_dataset_val",
            "true_label": 1,
            "validity_score": scores_by_split["novel_dataset_val"]["validity_scores"],
            "novelty_score": scores_by_split["novel_dataset_val"]["novelty_scores"],
        }),
    ], ignore_index=True)
    pred_dfs.append(df_data_bm)

    val_predictions_df = pd.concat(pred_dfs, ignore_index=True)
    val_pred_csv_path = out_dir / "val_predictions.csv"
    val_predictions_df.to_csv(val_pred_csv_path, index=False)
    print(f"  - Saved benchmark predictions: {val_pred_csv_path} ({len(val_predictions_df)} rows)")

    # B. val_scores_by_split.csv (Unique graphs: 1,855 rows with split and metadata)
    split_dfs = []
    for s_name in splits_to_eval:
        s_data = scores_by_split[s_name]
        df_sp = pd.DataFrame({
            "uid": s_data["uids"],
            "split": s_name,
            "validity_score": s_data["validity_scores"],
            "novelty_score": s_data["novelty_scores"],
        })
        split_dfs.append(df_sp)
    val_split_df = pd.concat(split_dfs, ignore_index=True)
    val_split_df = val_split_df.merge(master_df[["uid", "ood_kind", "family"]], on="uid", how="left")
    val_split_csv_path = out_dir / "val_scores_by_split.csv"
    val_split_df.to_csv(val_split_csv_path, index=False)
    print(f"  - Saved per-graph scores: {val_split_csv_path} ({len(val_split_df)} unique rows)")

    # C. val_toxicity_predictions.csv (Tox21 predictions preserving labels and NaNs)
    tox_task_names = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]
    tox_rows = []
    for s_name in splits_to_eval:
        s_data = scores_by_split[s_name]
        for i, uid in enumerate(s_data["uids"]):
            row_dict: Dict[str, Any] = {"uid": uid, "split": s_name}
            for t_idx, t_name in enumerate(tox_task_names):
                row_dict[f"{t_name}_label"] = float(s_data["toxicity_labels"][i, t_idx])
                row_dict[f"{t_name}_prob"] = float(s_data["toxicity_probs"][i, t_idx])
            tox_rows.append(row_dict)
    val_tox_df = pd.DataFrame(tox_rows)
    val_tox_csv_path = out_dir / "val_toxicity_predictions.csv"
    val_tox_df.to_csv(val_tox_csv_path, index=False)
    print(f"  - Saved toxicity predictions: {val_tox_csv_path} ({len(val_tox_df)} rows, NaNs preserved)")

    # D. val_metrics.json
    val_metrics_path = out_dir / "val_metrics.json"
    with open(val_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  - Saved metrics report: {val_metrics_path}")

    # E. evaluation_config.json
    eval_config = {
        "step": 5,
        "phase": "validation_evaluation",
        "toxicity_checkpoint": toxicity_checkpoint,
        "validity_checkpoint": validity_checkpoint,
        "novelty_estimator_path": novelty_estimator_path,
        "embedding_dimension": tox_model.embed_dim,
        "score_directions": {
            "validity_score": "Higher means more corrupted / invalid (positive class = corrupted)",
            "novelty_score": "Higher means more novel / farther from ID training manifold (positive class = novel)",
        },
        "score_bounds": {
            "validity_score": "[0.0, 1.0] (Sigmoid probability output)",
            "novelty_score": "[0.0, +inf) (Regularized Mahalanobis distance)",
        },
        "split_counts": {s: len(scores_by_split[s]["uids"]) for s in splits_to_eval},
        "test_protection_status": "Strictly frozen. No test split loaded or inspected during validation evaluation.",
    }
    config_path = out_dir / "evaluation_config.json"
    with open(config_path, "w") as f:
        json.dump(eval_config, f, indent=2)
    print(f"  - Saved evaluation configuration: {config_path}")

    print("\nValidation evaluation complete.")
    return {
        "metrics": metrics,
        "val_predictions_path": str(val_pred_csv_path),
        "val_split_scores_path": str(val_split_csv_path),
        "val_toxicity_path": str(val_tox_csv_path),
        "metrics_path": str(val_metrics_path),
        "config_path": str(config_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dual-Axis OOD Validation Evaluation.")
    parser.add_argument("--data_dir", type=str, default="processed_v3")
    parser.add_argument("--output_dir", type=str, default="predictions")
    parser.add_argument("--toxicity_checkpoint", type=str, default="checkpoints/toxicity_gine/best_model.pt")
    parser.add_argument("--validity_checkpoint", type=str, default="checkpoints/validity_gine/best_validity_model.pt")
    parser.add_argument("--novelty_estimator_path", type=str, default="checkpoints/novelty_mahal/estimator.npz")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    run_validation_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        toxicity_checkpoint=args.toxicity_checkpoint,
        validity_checkpoint=args.validity_checkpoint,
        novelty_estimator_path=args.novelty_estimator_path,
        batch_size=args.batch_size,
        device=args.device,
    )
