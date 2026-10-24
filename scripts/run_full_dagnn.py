"""
scripts/run_full_dagnn.py
=========================
Full training, calibration, and validation evaluation runner for DAG-NN (Track A).

Protocol Rules Enforced:
1. Training: id_train ONLY (5,432 graphs).
2. Early Stopping: Evaluated on id_val loss (patience=10, max_epochs=60).
3. Calibration: Full calibration fitted strictly on all 5,432 id_train graphs.
4. Validation Splits: id_val (679), corrupt_val (543), novel_scaffold_val (408), novel_dataset_val (225).
   ZERO test splits loaded or evaluated.
5. Paired Output: Exports validity_score and novelty_score for every graph in validation.
   single_score left blank per dual-axis protocol.
6. Checkpoint: Saves weights, full calibration buffers, is_calibrated flag, epoch, and seed.
7. Evaluates Tasks: V1, V2, N, NM (matched controls), X (cause separation), and Task D (3-class decision).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dual_axis_gnn import DAGNN, collate_graphs
from evaluation.metrics import compute_all_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute standard AUROC, AUPR, and FPR95 via evaluation.metrics."""
    res = compute_all_metrics(y_true, y_score)
    return {"auroc": res["auroc"], "aupr": res["aupr_out"], "fpr95": res["fpr95"]}


def main():
    parser = argparse.ArgumentParser(description="Run Full DAG-NN Training and Validation Pilot.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (approved protocol: 0..4).")
    parser.add_argument("--max_epochs", type=int, default=60, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on id_val loss.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device.")
    args = parser.parse_args()

    print("=" * 80)
    print(f"DAG-NN FULL RUNNER: TRACK A, SEED {args.seed} (VALIDATION PILOT)")
    print("=" * 80)

    set_seed(args.seed)
    device = torch.device(args.device)

    # -------------------------------------------------------------------------
    # 1. Paths and Safety Preflight
    # -------------------------------------------------------------------------
    graphs_dir = PROJECT_ROOT / "processed_v3" / "graphs"
    addendum_dir = PROJECT_ROOT / "processed_v3_addendum" / "processed_v3_addendum"
    labels_path = addendum_dir / "tables" / "dual_axis_labels.csv"
    matched_path = addendum_dir / "tables" / "matched_controls_filtered.csv"

    id_train_path = graphs_dir / "id_train.pt"
    id_val_path = graphs_dir / "id_val.pt"
    corrupt_val_path = graphs_dir / "corrupt_val.pt"
    nov_scaf_val_path = graphs_dir / "novel_scaffold_val.pt"
    nov_data_val_path = graphs_dir / "novel_dataset_val.pt"

    for p in [id_train_path, id_val_path, corrupt_val_path, nov_scaf_val_path, nov_data_val_path, labels_path]:
        assert p.exists(), f"Required path missing: {p}"

    pred_dir = PROJECT_ROOT / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_csv_path = pred_dir / f"DAGNN__trackA__seed{args.seed}__val.csv"

    ckpt_dir = PROJECT_ROOT / "checkpoints" / "dagnn_trackA"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"dagnn_trackA_seed{args.seed}.pt"

    results_dir = PROJECT_ROOT / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = results_dir / "dagnn_validation_evaluation_summary.csv"

    # Pre-flight overwrite safety
    if pred_csv_path.exists():
        print(f"[STOP] Target prediction file already exists: {pred_csv_path}")
        sys.exit(1)
    if ckpt_path.exists():
        print(f"[STOP] Target checkpoint file already exists: {ckpt_path}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. Data Loading (Training & Validation Only; Test Sets Never Loaded)
    # -------------------------------------------------------------------------
    print("\n[DATA LOAD] Loading training set (id_train)...")
    train_graphs = torch.load(id_train_path, weights_only=True)
    print(f"  Loaded {len(train_graphs)} training graphs.")

    print("[DATA LOAD] Loading validation sets...")
    val_id_graphs = torch.load(id_val_path, weights_only=True)
    val_corrupt_graphs = torch.load(corrupt_val_path, weights_only=True)
    val_nov_scaf_graphs = torch.load(nov_scaf_val_path, weights_only=True)
    val_nov_data_graphs = torch.load(nov_data_val_path, weights_only=True)

    print(f"  id_val:             {len(val_id_graphs):>5} graphs")
    print(f"  corrupt_val:        {len(val_corrupt_graphs):>5} graphs")
    print(f"  novel_scaffold_val: {len(val_nov_scaf_graphs):>5} graphs")
    print(f"  novel_dataset_val:  {len(val_nov_data_graphs):>5} graphs")
    total_val_graphs = len(val_id_graphs) + len(val_corrupt_graphs) + len(val_nov_scaf_graphs) + len(val_nov_data_graphs)
    print(f"  Total Validation:   {total_val_graphs:>5} graphs")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Training
    # -------------------------------------------------------------------------
    model = DAGNN(
        in_node_dim=30,
        in_edge_dim=6,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        num_tasks=12,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    batch_size = args.batch_size
    n_train_batches = math.ceil(len(train_graphs) / batch_size)
    n_val_batches = math.ceil(len(val_id_graphs) / batch_size)

    print(f"\n[TRAINING] Starting training (max_epochs={args.max_epochs}, patience={args.patience})...")
    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    patience_counter = 0

    t_train_start = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        # Shuffle training graphs per epoch
        perm = np.random.permutation(len(train_graphs))
        shuffled_train = [train_graphs[i] for i in perm]

        model.train()
        train_losses = []
        for b_idx in range(n_train_batches):
            batch_slice = shuffled_train[b_idx * batch_size : (b_idx + 1) * batch_size]
            collated = collate_graphs(batch_slice)
            optimizer.zero_grad()
            loss_dict = model.forward_train(collated)
            loss = loss_dict["loss_total"]
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate validation loss on id_val for early stopping
        model.eval()
        val_losses = []
        with torch.no_grad():
            for b_idx in range(n_val_batches):
                batch_slice = val_id_graphs[b_idx * batch_size : (b_idx + 1) * batch_size]
                collated = collate_graphs(batch_slice)
                loss_dict = model.forward_train(collated)
                val_losses.append(loss_dict["loss_total"].item())

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            improved = "*"
        else:
            patience_counter += 1
            improved = ""

        if epoch % 5 == 0 or improved or epoch == 1:
            print(f"  Epoch {epoch:2d}/{args.max_epochs:2d} | Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} {improved}")

        if patience_counter >= args.patience:
            print(f"  [EARLY STOP] Stopping early at epoch {epoch}. Best epoch was {best_epoch} (Val Loss: {best_val_loss:.4f}).")
            break

    t_train_elapsed = time.perf_counter() - t_train_start
    print(f"[TRAINING COMPLETE] Trained in {t_train_elapsed:.2f} s. Restoring weights from best epoch ({best_epoch})...")
    model.load_state_dict(best_state_dict)

    # -------------------------------------------------------------------------
    # 4. Full Calibration on All 5,432 id_train Graphs
    # -------------------------------------------------------------------------
    print(f"\n[CALIBRATION] Fitting full calibration strictly on all {len(train_graphs)} id_train graphs...")
    t_cal_start = time.perf_counter()
    model.fit_calibration(train_graphs, max_samples=None)
    t_cal_elapsed = time.perf_counter() - t_cal_start

    print(f"  Calibration completed in {t_cal_elapsed:.2f} s")
    print(f"  calib_mu_node:    {model.calib_mu_node.item():.4f}")
    print(f"  calib_sigma_node: {model.calib_sigma_node.item():.4f}")
    print(f"  calib_mu_edge:    {model.calib_mu_edge.item():.4f}")
    print(f"  calib_sigma_edge: {model.calib_sigma_edge.item():.4f}")
    print(f"  novelty_mu norm:  {torch.norm(model.novelty_mu).item():.4f}")

    # -------------------------------------------------------------------------
    # 5. Checkpoint Persistence
    # -------------------------------------------------------------------------
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "is_calibrated": model.is_calibrated,
        "epoch": best_epoch,
        "seed": args.seed,
        "hparams": {
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.1,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "best_val_loss": best_val_loss,
        },
        "calib_stats": {
            "calib_mu_node": model.calib_mu_node.item(),
            "calib_sigma_node": model.calib_sigma_node.item(),
            "calib_mu_edge": model.calib_mu_edge.item(),
            "calib_sigma_edge": model.calib_sigma_edge.item(),
            "novelty_mu_norm": torch.norm(model.novelty_mu).item(),
            "train_samples_calibrated": len(train_graphs),
        },
    }
    torch.save(checkpoint_data, ckpt_path)
    print(f"[CHECKPOINT] Saved full checkpoint to: {ckpt_path}")

    # -------------------------------------------------------------------------
    # 6. Full Validation Inference (All 1,855 Graphs)
    # -------------------------------------------------------------------------
    print("\n[INFERENCE] Evaluating all validation splits with deterministic Leave-One-Out (LOO)...")
    all_val_graphs = val_id_graphs + val_corrupt_graphs + val_nov_scaf_graphs + val_nov_data_graphs
    
    t_inf_start = time.perf_counter()
    pred_records = []
    
    for idx, g in enumerate(all_val_graphs, 1):
        scores = model.score_graph(g)
        pred_records.append({
            "uid": g["uid"],
            "model": "DAGNN",
            "track": "A",
            "seed": args.seed,
            "validity_score": scores["validity_score"],
            "novelty_score": scores["novelty_score"],
            "single_score": np.nan,  # Left blank per protocol
        })
        if idx % 500 == 0 or idx == len(all_val_graphs):
            print(f"  Scored {idx:4d}/{len(all_val_graphs)} graphs...")

    t_inf_elapsed = time.perf_counter() - t_inf_start
    print(f"  Inference finished in {t_inf_elapsed:.2f} s ({t_inf_elapsed / len(all_val_graphs) * 1000:.2f} ms/graph)")

    pred_df = pd.DataFrame(pred_records)

    # Export prediction CSV
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"[EXPORT] Wrote {len(pred_df)} predictions to: {pred_csv_path}")

    # -------------------------------------------------------------------------
    # 7. Post-Flight Pre-Evaluation Validation
    # -------------------------------------------------------------------------
    print("\n[VALIDATION CHECK] Running validate_predictions logic...")
    val_script_path = PROJECT_ROOT / "processed_v3_ready" / "processed_v3_ready" / "scripts" / "validate_predictions.py"
    sys.path.insert(0, str(val_script_path.parent))
    from validate_predictions import validate
    labels_df = pd.read_csv(labels_path, keep_default_na=False)

    val_errors, val_warnings = validate(labels_df, pred_df, "val", pred_csv_path.name)
    for w in val_warnings:
        print(f"  [WARNING] {w}")
    if val_errors:
        print(f"  [ERROR] Validation failed: {val_errors}")
        sys.exit(1)
    print("  [PASS] Pre-flight validation passed with 0 errors!")

    # -------------------------------------------------------------------------
    # 8. Evaluation of All Supported Tasks (V1, V2, N, NM, X, Task D)
    # -------------------------------------------------------------------------
    print("\n[EVALUATION] Computing benchmark metrics on non-excluded graphs...")
    # Merge predictions with authoritative dual_axis_labels.csv
    eval_df = labels_df[(labels_df["split"] == "val") & (labels_df["exclude"] == 0)].merge(
        pred_df[["uid", "validity_score", "novelty_score"]], on="uid", how="inner"
    )

    idd = eval_df[eval_df["ood_class"] == "id"]
    nov = eval_df[eval_df["ood_class"] == "novel"]
    inv = eval_df[eval_df["ood_class"] == "invalid"]
    nov_scaf = nov[nov["novelty_kind"] == "scaffold"]
    nov_data = nov[nov["novelty_kind"] == "dataset"]
    inv_a = inv[inv["corruption_family"] == "A"]
    inv_b = inv[inv["corruption_family"] == "B"]

    results = []

    def log_result(task: str, benchmark: str, score_col: str, pos_df: pd.DataFrame, neg_df: pd.DataFrame):
        y_true = np.r_[np.ones(len(pos_df)), np.zeros(len(neg_df))]
        y_score = np.r_[pos_df[score_col].to_numpy(float), neg_df[score_col].to_numpy(float)]
        m = compute_metrics(y_true, y_score)
        results.append({
            "model_name": "DAGNN",
            "seed": args.seed,
            "task": task,
            "benchmark": benchmark,
            "score_col": score_col,
            "auroc": m["auroc"],
            "aupr": m["aupr"],
            "fpr95": m["fpr95"],
            "n_pos": len(pos_df),
            "n_neg": len(neg_df),
            "n_total": len(pos_df) + len(neg_df),
        })

    # Task V1: Validity vs ID
    log_result("V1", "validity_family_a_val", "validity_score", inv_a, idd)
    log_result("V1", "validity_family_b_val", "validity_score", inv_b, idd)
    log_result("V1", "validity_family_ALL_val", "validity_score", inv, idd)

    # Task V2: Validity vs ID + Novel (ensures novel is not flagged invalid)
    all_valid = pd.concat([idd, nov], ignore_index=True)
    log_result("V2", "validity_family_a_vs_all_valid", "validity_score", inv_a, all_valid)
    log_result("V2", "validity_family_b_vs_all_valid", "validity_score", inv_b, all_valid)
    log_result("V2", "validity_ALL_vs_all_valid", "validity_score", inv, all_valid)

    # Task N: Novelty vs ID
    log_result("N", "novelty_scaffold_val", "novelty_score", nov_scaf, idd)
    log_result("N", "novelty_dataset_val", "novelty_score", nov_data, idd)

    # Task NM: Matched Controls
    if matched_path.exists():
        mc_df = pd.read_csv(matched_path)
        score_map = eval_df.set_index("uid")["novelty_score"]
        
        # Scaffold matched controls
        mc_scaf = mc_df[mc_df["target_set"] == "novel_scaffold_val"]
        scaf_t = mc_scaf[mc_scaf["target_uid"].isin(score_map.index) & mc_scaf["control_uid"].isin(score_map.index)]
        pos_scaf_mc = pd.DataFrame({"novelty_score": score_map.loc[scaf_t["target_uid"]].values})
        neg_scaf_mc = pd.DataFrame({"novelty_score": score_map.loc[scaf_t["control_uid"]].values})
        log_result("NM", "novelty_scaffold_matched_controls", "novelty_score", pos_scaf_mc, neg_scaf_mc)

        # Dataset matched controls
        mc_data = mc_df[mc_df["target_set"] == "novel_dataset_val"]
        data_t = mc_data[mc_data["target_uid"].isin(score_map.index) & mc_data["control_uid"].isin(score_map.index)]
        pos_data_mc = pd.DataFrame({"novelty_score": score_map.loc[data_t["target_uid"]].values})
        neg_data_mc = pd.DataFrame({"novelty_score": score_map.loc[data_t["control_uid"]].values})
        log_result("NM", "novelty_dataset_matched_controls", "novelty_score", pos_data_mc, neg_data_mc)

    # Task X: Cause Separation (can validity_score separate corrupted from novel?)
    log_result("X", "cause_sep_family_a_vs_novel", "validity_score", inv_a, nov)
    log_result("X", "cause_sep_family_b_vs_novel", "validity_score", inv_b, nov)
    log_result("X", "cause_sep_ALL_vs_novel", "validity_score", inv, nov)

    # -------------------------------------------------------------------------
    # 9. Task D: Joint 3-Class Decision Evaluation
    # -------------------------------------------------------------------------
    # Thresholds derived strictly from id_val (95th percentiles)
    tau_v = float(np.percentile(idd["validity_score"].to_numpy(), 95))
    tau_n = float(np.percentile(idd["novelty_score"].to_numpy(), 95))

    # Decision rule:
    # 0 = ID, 1 = Invalid, 2 = Novel
    def classify_d(row) -> int:
        if row["validity_score"] >= tau_v:
            return 1  # Invalid
        elif row["novelty_score"] >= tau_n:
            return 2  # Novel
        else:
            return 0  # In-Distribution

    eval_df["pred_class"] = eval_df.apply(classify_d, axis=1)
    class_map = {"id": 0, "invalid": 1, "novel": 2}
    eval_df["true_class"] = eval_df["ood_class"].map(class_map)

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(eval_df["true_class"].to_numpy(), eval_df["pred_class"].to_numpy()):
        cm[t, p] += 1
    # Per-class accuracies
    rec_id = cm[0, 0] / cm[0].sum()
    rec_inv = cm[1, 1] / cm[1].sum()
    rec_nov = cm[2, 2] / cm[2].sum()
    macro_recall = float((rec_id + rec_inv + rec_nov) / 3.0)

    print("\n" + "=" * 80)
    print("TASK D JOINT 3-CLASS DECISION MATRIX (id_val 95th Percentile Thresholds)")
    print("=" * 80)
    print(f"Thresholds: tau_validity = {tau_v:.4f}, tau_novelty = {tau_n:.4f}")
    print("Confusion Matrix (Rows: True [ID, Invalid, Novel], Cols: Pred [ID, Invalid, Novel]):")
    print(cm)
    print(f"Recall ID: {rec_id:.4f} | Recall Invalid: {rec_inv:.4f} | Recall Novel: {rec_nov:.4f}")
    print(f"Macro Recall / Balanced Accuracy: {macro_recall:.4f}")

    results.append({
        "model_name": "DAGNN",
        "seed": args.seed,
        "task": "Task_D",
        "benchmark": "joint_3class_decision",
        "score_col": "paired_validity_novelty",
        "auroc": macro_recall,  # Used here to store macro-recall
        "aupr": rec_inv,
        "fpr95": rec_nov,
        "n_pos": int(cm[1].sum() + cm[2].sum()),
        "n_neg": int(cm[0].sum()),
        "n_total": len(eval_df),
    })

    # Save summary table
    res_df = pd.DataFrame(results)
    res_df.to_csv(summary_csv_path, index=False)
    print(f"\n[SUMMARY] Saved full evaluation summary to: {summary_csv_path}")

    print("\n" + "=" * 80)
    print("DAG-NN VALIDATION BENCHMARK EVALUATION SUMMARY (SEED 0 PILOT)")
    print("=" * 80)
    display_cols = ["task", "benchmark", "score_col", "auroc", "aupr", "fpr95", "n_pos", "n_neg"]
    print(res_df[display_cols].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
