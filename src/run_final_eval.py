"""
Final Frozen Test Evaluation Runner for Dual-Axis OOD Detection.

STRICT PROTOCOL ENFORCEMENT:
- This script evaluates the frozen components on the test splits:
  * id_test (679 graphs)
  * corrupt_test (543 graphs: Fam A = 181, Fam B = 181, Fam C = 181)
  * novel_scaffold_test (347 graphs)
  * novel_dataset_test (198 graphs)
- Family C is evaluated for the FIRST AND ONLY time here.
- ZERO tuning, hyperparameter search, threshold calibration, or model adjustment
  may be performed based on the output of this script.
- Requires explicit confirmation flag `--authorize_test_evaluation`.
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
logger = logging.getLogger("run_final_eval")


def run_test_evaluation(
    data_dir: str = "processed_v3",
    output_dir: str = "predictions",
    toxicity_checkpoint: str = "checkpoints/toxicity_gine/best_model.pt",
    validity_checkpoint: str = "checkpoints/validity_gine/best_validity_model.pt",
    novelty_estimator_path: str = "checkpoints/novelty_mahal/estimator.npz",
    batch_size: int = 64,
    device: str = "cpu",
    authorize: bool = False,
) -> Dict[str, Any]:
    """
    Run final test evaluation on frozen models.
    """
    if not authorize:
        print("\n[BLOCKED] Final test evaluation was called without authorization.")
        print("To run final evaluation on test sets (including Family C), specify:")
        print("    python -m src.run_final_eval --authorize_test_evaluation\n")
        sys.exit(1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    print("=" * 70)
    print("STEP 5: INTEGRATED MODEL EVALUATION -- FINAL FROZEN TEST SUITE")
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

    # 2. Score test splits
    print("\n[2/5] Scoring test splits...")
    splits_to_eval = [
        "id_test",
        "corrupt_test",
        "novel_scaffold_test",
        "novel_dataset_test",
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

    corrupt_test_meta = master_df[master_df["set_name"] == "corrupt_test"][["uid", "family"]].copy()
    corrupt_scores_df = pd.DataFrame({
        "uid": scores_by_split["corrupt_test"]["uids"],
        "validity_score": scores_by_split["corrupt_test"]["validity_scores"],
        "novelty_score": scores_by_split["corrupt_test"]["novelty_scores"],
    }).merge(corrupt_test_meta, on="uid")

    fam_a_uids = corrupt_scores_df[corrupt_scores_df["family"] == "A"]["uid"].tolist()
    fam_a_val_scores = corrupt_scores_df[corrupt_scores_df["family"] == "A"]["validity_score"].to_numpy()
    fam_a_nov_scores = corrupt_scores_df[corrupt_scores_df["family"] == "A"]["novelty_score"].to_numpy()

    fam_b_uids = corrupt_scores_df[corrupt_scores_df["family"] == "B"]["uid"].tolist()
    fam_b_val_scores = corrupt_scores_df[corrupt_scores_df["family"] == "B"]["validity_score"].to_numpy()
    fam_b_nov_scores = corrupt_scores_df[corrupt_scores_df["family"] == "B"]["novelty_score"].to_numpy()

    fam_c_uids = corrupt_scores_df[corrupt_scores_df["family"] == "C"]["uid"].tolist()
    fam_c_val_scores = corrupt_scores_df[corrupt_scores_df["family"] == "C"]["validity_score"].to_numpy()
    fam_c_nov_scores = corrupt_scores_df[corrupt_scores_df["family"] == "C"]["novelty_score"].to_numpy()

    # Fast UID-to-score mappings
    id_test_dict = dict(zip(scores_by_split["id_test"]["uids"], scores_by_split["id_test"]["novelty_scores"]))
    scaff_test_dict = dict(zip(scores_by_split["novel_scaffold_test"]["uids"], scores_by_split["novel_scaffold_test"]["novelty_scores"]))
    data_test_dict = dict(zip(scores_by_split["novel_dataset_test"]["uids"], scores_by_split["novel_dataset_test"]["novelty_scores"]))

    # Matched controls extraction
    scaff_matched = controls_df[(controls_df["target_set"] == "novel_scaffold_test") & (controls_df["matched"] == True)].copy()
    scaff_matched["target_score"] = scaff_matched["target_uid"].map(scaff_test_dict)
    scaff_matched["control_score"] = scaff_matched["control_uid"].map(id_test_dict)

    data_matched = controls_df[(controls_df["target_set"] == "novel_dataset_test") & (controls_df["matched"] == True)].copy()
    data_matched["target_score"] = data_matched["target_uid"].map(data_test_dict)
    data_matched["control_score"] = data_matched["control_uid"].map(id_test_dict)

    # 4. Compute Metrics
    print("\n[4/5] Computing Benchmark Performance Metrics on TEST Splits...")
    metrics: Dict[str, Any] = {}

    # --- Validity Test Metrics ---
    # Validity All: id_test (0) vs corrupt_test (1)
    y_val_all = [0] * len(scores_by_split["id_test"]["uids"]) + [1] * len(scores_by_split["corrupt_test"]["uids"])
    s_val_all = np.concatenate([scores_by_split["id_test"]["validity_scores"], scores_by_split["corrupt_test"]["validity_scores"]])
    metrics["validity_all_test"] = compute_ood_metrics(y_val_all, s_val_all)

    # Validity Family A: id_test (0) vs corrupt_test Fam A (1)
    y_val_a = [0] * len(scores_by_split["id_test"]["uids"]) + [1] * len(fam_a_uids)
    s_val_a = np.concatenate([scores_by_split["id_test"]["validity_scores"], fam_a_val_scores])
    metrics["validity_family_a_test"] = compute_ood_metrics(y_val_a, s_val_a)

    # Validity Family B: id_test (0) vs corrupt_test Fam B (1)
    y_val_b = [0] * len(scores_by_split["id_test"]["uids"]) + [1] * len(fam_b_uids)
    s_val_b = np.concatenate([scores_by_split["id_test"]["validity_scores"], fam_b_val_scores])
    metrics["validity_family_b_test"] = compute_ood_metrics(y_val_b, s_val_b)

    # Validity Family C (Held-out Carbon -> Oxygen swap): id_test (0) vs corrupt_test Fam C (1)
    y_val_c = [0] * len(scores_by_split["id_test"]["uids"]) + [1] * len(fam_c_uids)
    s_val_c = np.concatenate([scores_by_split["id_test"]["validity_scores"], fam_c_val_scores])
    metrics["validity_family_c_test"] = compute_ood_metrics(y_val_c, s_val_c)

    # --- Novelty Test Metrics ---
    # Novel Scaffold (Raw): id_test (0) vs novel_scaffold_test (1)
    y_scaff_raw = [0] * len(scores_by_split["id_test"]["uids"]) + [1] * len(scores_by_split["novel_scaffold_test"]["uids"])
    s_scaff_raw = np.concatenate([scores_by_split["id_test"]["novelty_scores"], scores_by_split["novel_scaffold_test"]["novelty_scores"]])
    metrics["novelty_scaffold_raw_test"] = compute_ood_metrics(y_scaff_raw, s_scaff_raw)

    # Novel Scaffold (Matched Controls)
    y_scaff_match = [0] * len(scaff_matched) + [1] * len(scaff_matched)
    s_scaff_match = np.concatenate([scaff_matched["control_score"].to_numpy(), scaff_matched["target_score"].to_numpy()])
    metrics["novelty_scaffold_matched_test"] = compute_ood_metrics(y_scaff_match, s_scaff_match)

    # Novel Dataset (Raw): id_test (0) vs novel_dataset_test (1)
    y_data_raw = [0] * len(scores_by_split["id_test"]["uids"]) + [1] * len(scores_by_split["novel_dataset_test"]["uids"])
    s_data_raw = np.concatenate([scores_by_split["id_test"]["novelty_scores"], scores_by_split["novel_dataset_test"]["novelty_scores"]])
    metrics["novelty_dataset_raw_test"] = compute_ood_metrics(y_data_raw, s_data_raw)

    # Novel Dataset (Matched Controls)
    y_data_match = [0] * len(data_matched) + [1] * len(data_matched)
    s_data_match = np.concatenate([data_matched["control_score"].to_numpy(), data_matched["target_score"].to_numpy()])
    metrics["novelty_dataset_matched_test"] = compute_ood_metrics(y_data_match, s_data_match)

    # Print Metrics Table
    print("\n" + "-" * 75)
    print(f"{'Benchmark / Evaluation Setting (TEST)':<36} | {'AUROC':>7} | {'AUPR':>7} | {'FPR95':>7} | {'N (Neg/Pos)':>11}")
    print("-" * 75)
    for k, m in metrics.items():
        n_str = f"{m['n_neg']}/{m['n_pos']}"
        print(f"{k:<36} | {m['auroc']:>7.4f} | {m['aupr']:>7.4f} | {m['fpr95']:>7.4f} | {n_str:>11}")
    print("-" * 75)

    # 5. Export Prediction Files
    print("\n[5/5] Exporting standardized TEST prediction files...")

    # A. test_predictions.csv (agreed schema: uid, benchmark/split, true_label, validity_score, novelty_score)
    pred_dfs: List[pd.DataFrame] = []

    # Validity benchmark rows
    df_val_bm = pd.concat([
        pd.DataFrame({
            "uid": scores_by_split["id_test"]["uids"],
            "benchmark/split": "validity/id_test",
            "true_label": 0,
            "validity_score": scores_by_split["id_test"]["validity_scores"],
            "novelty_score": scores_by_split["id_test"]["novelty_scores"],
        }),
        pd.DataFrame({
            "uid": fam_a_uids,
            "benchmark/split": "validity/corrupt_test_fam_a",
            "true_label": 1,
            "validity_score": fam_a_val_scores,
            "novelty_score": fam_a_nov_scores,
        }),
        pd.DataFrame({
            "uid": fam_b_uids,
            "benchmark/split": "validity/corrupt_test_fam_b",
            "true_label": 1,
            "validity_score": fam_b_val_scores,
            "novelty_score": fam_b_nov_scores,
        }),
        pd.DataFrame({
            "uid": fam_c_uids,
            "benchmark/split": "validity/corrupt_test_fam_c",
            "true_label": 1,
            "validity_score": fam_c_val_scores,
            "novelty_score": fam_c_nov_scores,
        }),
    ], ignore_index=True)
    pred_dfs.append(df_val_bm)

    # Novelty scaffold benchmark rows
    df_scaff_bm = pd.concat([
        pd.DataFrame({
            "uid": scores_by_split["id_test"]["uids"],
            "benchmark/split": "novelty_scaffold/id_test",
            "true_label": 0,
            "validity_score": scores_by_split["id_test"]["validity_scores"],
            "novelty_score": scores_by_split["id_test"]["novelty_scores"],
        }),
        pd.DataFrame({
            "uid": scores_by_split["novel_scaffold_test"]["uids"],
            "benchmark/split": "novelty_scaffold/novel_scaffold_test",
            "true_label": 1,
            "validity_score": scores_by_split["novel_scaffold_test"]["validity_scores"],
            "novelty_score": scores_by_split["novel_scaffold_test"]["novelty_scores"],
        }),
    ], ignore_index=True)
    pred_dfs.append(df_scaff_bm)

    # Novelty dataset benchmark rows
    df_data_bm = pd.concat([
        pd.DataFrame({
            "uid": scores_by_split["id_test"]["uids"],
            "benchmark/split": "novelty_dataset/id_test",
            "true_label": 0,
            "validity_score": scores_by_split["id_test"]["validity_scores"],
            "novelty_score": scores_by_split["id_test"]["novelty_scores"],
        }),
        pd.DataFrame({
            "uid": scores_by_split["novel_dataset_test"]["uids"],
            "benchmark/split": "novelty_dataset/novel_dataset_test",
            "true_label": 1,
            "validity_score": scores_by_split["novel_dataset_test"]["validity_scores"],
            "novelty_score": scores_by_split["novel_dataset_test"]["novelty_scores"],
        }),
    ], ignore_index=True)
    pred_dfs.append(df_data_bm)

    test_predictions_df = pd.concat(pred_dfs, ignore_index=True)
    test_pred_csv_path = out_dir / "test_predictions.csv"
    test_predictions_df.to_csv(test_pred_csv_path, index=False)
    print(f"  - Saved benchmark test predictions: {test_pred_csv_path} ({len(test_predictions_df)} rows)")

    # B. test_scores_by_split.csv
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
    test_split_df = pd.concat(split_dfs, ignore_index=True)
    test_split_df = test_split_df.merge(master_df[["uid", "ood_kind", "family"]], on="uid", how="left")
    test_split_csv_path = out_dir / "test_scores_by_split.csv"
    test_split_df.to_csv(test_split_csv_path, index=False)
    print(f"  - Saved per-graph test scores: {test_split_csv_path} ({len(test_split_df)} unique rows)")

    # C. test_toxicity_predictions.csv
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
    test_tox_df = pd.DataFrame(tox_rows)
    test_tox_csv_path = out_dir / "test_toxicity_predictions.csv"
    test_tox_df.to_csv(test_tox_csv_path, index=False)
    print(f"  - Saved test toxicity predictions: {test_tox_csv_path} ({len(test_tox_df)} rows, NaNs preserved)")

    # D. test_metrics.json
    test_metrics_path = out_dir / "test_metrics.json"
    with open(test_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  - Saved test metrics report: {test_metrics_path}")

    # E. test_evaluation_config.json
    test_eval_config = {
        "step": 5,
        "phase": "final_frozen_test_evaluation",
        "toxicity_checkpoint": toxicity_checkpoint,
        "validity_checkpoint": validity_checkpoint,
        "novelty_estimator_path": novelty_estimator_path,
        "embedding_dimension": tox_model.embed_dim,
        "split_counts": {s: len(scores_by_split[s]["uids"]) for s in splits_to_eval},
        "family_c_graphs_evaluated": len(fam_c_uids),
        "post_evaluation_tuning": "STRICTLY FORBIDDEN. All model weights and parameters remain frozen.",
    }
    config_path = out_dir / "test_evaluation_config.json"
    with open(config_path, "w") as f:
        json.dump(test_eval_config, f, indent=2)
    print(f"  - Saved test evaluation configuration: {config_path}")

    print("\nFinal test evaluation complete.")
    return {
        "metrics": metrics,
        "test_predictions_path": str(test_pred_csv_path),
        "test_split_scores_path": str(test_split_csv_path),
        "test_toxicity_path": str(test_tox_csv_path),
        "metrics_path": str(test_metrics_path),
        "config_path": str(config_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Final Frozen Test Evaluation.")
    parser.add_argument("--authorize_test_evaluation", action="store_true", help="Authorize final test evaluation.")
    parser.add_argument("--data_dir", type=str, default="processed_v3")
    parser.add_argument("--output_dir", type=str, default="predictions")
    parser.add_argument("--toxicity_checkpoint", type=str, default="checkpoints/toxicity_gine/best_model.pt")
    parser.add_argument("--validity_checkpoint", type=str, default="checkpoints/validity_gine/best_validity_model.pt")
    parser.add_argument("--novelty_estimator_path", type=str, default="checkpoints/novelty_mahal/estimator.npz")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    run_test_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        toxicity_checkpoint=args.toxicity_checkpoint,
        validity_checkpoint=args.validity_checkpoint,
        novelty_estimator_path=args.novelty_estimator_path,
        batch_size=args.batch_size,
        device=args.device,
        authorize=args.authorize_test_evaluation,
    )
