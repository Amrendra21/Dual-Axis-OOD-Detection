"""
scripts/run_id_reference_replacement_pilot.py
=============================================
Controlled ID-Reference Replacement Diagnostic Pilot for Scaffold Novelty.

Research Question:
Does the measured DAG-NN scaffold-novelty AUROC change when the ID reference
population is restricted to the subset of ID validation molecules whose scaffolds
overlap with the ID training set?

Evaluates three reference conditions against the identical 408 active novel_scaffold_val molecules:
1. Full active id_val reference (N=678)
2. Generic Murcko framework-overlap reference (N=190)
3. Ring-free / acyclic id_val reference (N=169)

Evaluates under two explicit mathematical paradigms:
- Paradigm A: Frozen id_train density estimator (evaluation cohort subsetting)
- Paradigm B: Re-estimated density estimator fitted strictly on the chosen reference subset

Safety Protocol Enforced:
- Frozen Seed-0 weights (checkpoints/dagnn_trackA/dagnn_trackA_seed0.pt)
- Validation partitions only; ZERO test partition access
- Read-only preservation of existing files; saves to dedicated pilot paths
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dual_axis_gnn import DAGNN, collate_graphs
from evaluation.metrics import compute_all_metrics


def get_reference_subsets(
    labels_path: Path,
    master_path: Path
) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """Identify active validation UIDs and the three reference cohorts."""
    labels_df = pd.read_csv(labels_path)
    master_df = pd.read_csv(master_path)

    val_labels = labels_df[(labels_df["split"] == "val") & (labels_df["exclude"] == 0)]
    id_val_active_uids = set(val_labels[val_labels["ood_class"] == "id"]["uid"])
    novel_scaf_active_uids = set(
        val_labels[(val_labels["ood_class"] == "novel") & (val_labels["novelty_kind"] == "scaffold")]["uid"]
    )

    master_id_val = master_df[master_df["uid"].isin(id_val_active_uids)].set_index("uid")
    master_id_train = master_df[master_df["set_name"] == "id_train"]

    # 1. Ring-free / acyclic active id_val
    acyclic_uids = set(master_id_val[master_id_val["is_acyclic"] == True].index)

    # 2. Generic Murcko framework overlap
    train_generic = set()
    for smi in master_id_train["scaffold"].dropna():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_generic.add(Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(mol)))

    generic_overlap_uids = set()
    for uid, row in master_id_val.iterrows():
        if row["is_acyclic"]:
            continue
        smi = row["scaffold"]
        if pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol:
            gen = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(mol))
            if gen in train_generic:
                generic_overlap_uids.add(uid)

    return id_val_active_uids, generic_overlap_uids, acyclic_uids, novel_scaf_active_uids


def extract_embeddings(
    model: DAGNN,
    graphs_id_val: List[Dict],
    graphs_nov_val: List[Dict]
) -> Dict[str, np.ndarray]:
    """Extract Layer 3 mean-pooled embeddings z_G deterministically."""
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for g in graphs_id_val + graphs_nov_val:
            u = g["uid"]
            batch = collate_graphs([g])
            _, z_G = model.embed_nodes_and_graph(batch)
            embeddings[u] = z_G[0].cpu().numpy()
    return embeddings


def reestimate_mahalanobis(
    ref_uids: Set[str],
    embeddings: Dict[str, np.ndarray],
    shrinkage: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Fit centroid and regularized precision matrix strictly on reference embeddings."""
    Z_ref = np.array([embeddings[u] for u in ref_uids])
    d = Z_ref.shape[1]
    mu = Z_ref.mean(axis=0)
    Z_cent = Z_ref - mu
    cov = (Z_cent.T @ Z_cent) / max(len(Z_ref) - 1, 1)

    trace_cov = np.trace(cov) / d
    cov_reg = (1.0 - shrinkage) * cov + shrinkage * trace_cov * np.eye(d)
    prec = np.linalg.pinv(cov_reg)

    eigvals = np.linalg.eigvalsh(cov_reg)
    cond = float(eigvals[-1] / max(eigvals[0], 1e-12))

    diag = {
        "n_samples": len(Z_ref),
        "d_features": d,
        "cond_number": cond,
        "min_eigval": float(eigvals[0]),
        "max_eigval": float(eigvals[-1]),
        "mean_trace": float(trace_cov),
    }
    return mu, prec, diag


def main():
    parser = argparse.ArgumentParser(description="Run Controlled ID-Reference Replacement Pilot.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dagnn_trackA/dagnn_trackA_seed0.pt")
    parser.add_argument("--output_csv", type=str, default="predictions/pilot_id_reference_replacement_predictions.csv")
    parser.add_argument("--output_report", type=str, default="evaluation/results/dagnn_controlled_id_reference_replacement_pilot_report.md")
    args = parser.parse_args()

    print("=== STARTING CONTROLLED ID-REFERENCE REPLACEMENT PILOT ===")

    labels_path = PROJECT_ROOT / "processed_v3_addendum/processed_v3_addendum/tables/dual_axis_labels.csv"
    master_path = PROJECT_ROOT / "processed_v3/tables/master_table.csv"
    ckpt_path = PROJECT_ROOT / args.checkpoint

    # Identify subsets
    id_full, id_gen, id_acy, nov_scaf = get_reference_subsets(labels_path, master_path)
    print(f"Cohort Counts: Full ID={len(id_full)}, Generic Overlap ID={len(id_gen)}, Acyclic ID={len(id_acy)}, Novel Scaffold Val={len(nov_scaf)}")

    # Load frozen model checkpoint
    print(f"Loading frozen checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DAGNN(in_node_dim=30, in_edge_dim=6, hidden_dim=128, num_layers=3, dropout=0.1, num_tasks=12)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load graphs
    g_id_val = torch.load(PROJECT_ROOT / "processed_v3/graphs/id_val.pt", weights_only=False)
    g_nov_val = torch.load(PROJECT_ROOT / "processed_v3/graphs/novel_scaffold_val.pt", weights_only=False)

    embeddings = extract_embeddings(model, g_id_val, g_nov_val)

    # Load existing baseline scores for Paradigm A
    preds_seed0 = pd.read_csv(PROJECT_ROOT / "predictions/DAGNN__trackA__seed0__val.csv").set_index("uid")
    scores_par_a = preds_seed0["novelty_score"]

    # Re-estimate estimators for Paradigm B
    mu_full, prec_full, diag_full = reestimate_mahalanobis(id_full, embeddings)
    mu_gen, prec_gen, diag_gen = reestimate_mahalanobis(id_gen, embeddings)
    mu_acy, prec_acy, diag_acy = reestimate_mahalanobis(id_acy, embeddings)

    # Calculate Paradigm B scores
    def calc_b(u: str, mu: np.ndarray, prec: np.ndarray) -> float:
        z = embeddings[u] - mu
        return float(z @ prec @ z)

    prediction_records = []
    # Save scores for all evaluated UIDs across all conditions
    all_evaluated_uids = id_full.union(nov_scaf)
    for u in sorted(all_evaluated_uids):
        is_ood = 1 if u in nov_scaf else 0
        ref_membership = []
        if u in id_full:
            ref_membership.append("full_id")
        if u in id_gen:
            ref_membership.append("generic_overlap_id")
        if u in id_acy:
            ref_membership.append("acyclic_id")
        if u in nov_scaf:
            ref_membership.append("novel_scaffold_target")

        score_a = float(scores_par_a.loc[u]) if u in scores_par_a.index else np.nan
        score_b_full = calc_b(u, mu_full, prec_full)
        score_b_gen = calc_b(u, mu_gen, prec_gen)
        score_b_acy = calc_b(u, mu_acy, prec_acy)

        prediction_records.append({
            "uid": u,
            "ground_truth": is_ood,
            "membership": ";".join(ref_membership),
            "score_paradigm_a_frozen": score_a,
            "score_paradigm_b_ref_full": score_b_full,
            "score_paradigm_b_ref_generic": score_b_gen,
            "score_paradigm_b_ref_acyclic": score_b_acy,
        })

    pred_df = pd.DataFrame(prediction_records)
    out_csv = PROJECT_ROOT / args.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_csv, index=False)
    print(f"Saved pilot predictions to: {out_csv}")

    # Compute metrics for both paradigms
    def eval_pair(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Dict[str, float]:
        y_true = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
        y_score = np.r_[pos_scores, neg_scores]
        return compute_all_metrics(y_true, y_score)

    nov_list = sorted(nov_scaf)
    pos_a = np.array([scores_par_a.loc[u] for u in nov_list])
    pos_b_full = np.array([calc_b(u, mu_full, prec_full) for u in nov_list])
    pos_b_gen = np.array([calc_b(u, mu_gen, prec_gen) for u in nov_list])
    pos_b_acy = np.array([calc_b(u, mu_acy, prec_acy) for u in nov_list])

    # Paradigm A evaluations
    m_a_full = eval_pair(pos_a, np.array([scores_par_a.loc[u] for u in sorted(id_full)]))
    m_a_gen = eval_pair(pos_a, np.array([scores_par_a.loc[u] for u in sorted(id_gen)]))
    m_a_acy = eval_pair(pos_a, np.array([scores_par_a.loc[u] for u in sorted(id_acy)]))

    # Paradigm B evaluations
    m_b_full = eval_pair(pos_b_full, np.array([calc_b(u, mu_full, prec_full) for u in sorted(id_full)]))
    m_b_gen = eval_pair(pos_b_gen, np.array([calc_b(u, mu_gen, prec_gen) for u in sorted(id_gen)]))
    m_b_acy = eval_pair(pos_b_acy, np.array([calc_b(u, mu_acy, prec_acy) for u in sorted(id_acy)]))

    print("\n=== PILOT RESULTS SUMMARY ===")
    print(f"Condition 1 (Full active id_val, N_pos=408, N_neg=678):")
    print(f"  Paradigm A (Frozen id_train Estimator): AUROC={m_a_full['auroc']:.4f}, AUPR={m_a_full['aupr_out']:.4f}, FPR95={m_a_full['fpr95']:.4f}")
    print(f"  Paradigm B (Re-estimated Estimator):    AUROC={m_b_full['auroc']:.4f}, AUPR={m_b_full['aupr_out']:.4f}, FPR95={m_b_full['fpr95']:.4f} (Cond={diag_full['cond_number']:.1f})")

    print(f"\nCondition 2 (Generic Murcko Overlap, N_pos=408, N_neg=190):")
    print(f"  Paradigm A (Frozen id_train Estimator): AUROC={m_a_gen['auroc']:.4f}, AUPR={m_a_gen['aupr_out']:.4f}, FPR95={m_a_gen['fpr95']:.4f} (Diff={m_a_gen['auroc'] - m_a_full['auroc']:+.4f})")
    print(f"  Paradigm B (Re-estimated Estimator):    AUROC={m_b_gen['auroc']:.4f}, AUPR={m_b_gen['aupr_out']:.4f}, FPR95={m_b_gen['fpr95']:.4f} (Diff={m_b_gen['auroc'] - m_b_full['auroc']:+.4f}, Cond={diag_gen['cond_number']:.1f})")

    print(f"\nCondition 3 (Ring-Free / Acyclic, N_pos=408, N_neg=169):")
    print(f"  Paradigm A (Frozen id_train Estimator): AUROC={m_a_acy['auroc']:.4f}, AUPR={m_a_acy['aupr_out']:.4f}, FPR95={m_a_acy['fpr95']:.4f} (Diff={m_a_acy['auroc'] - m_a_full['auroc']:+.4f})")
    print(f"  Paradigm B (Re-estimated Estimator):    AUROC={m_b_acy['auroc']:.4f}, AUPR={m_b_acy['aupr_out']:.4f}, FPR95={m_b_acy['fpr95']:.4f} (Diff={m_b_acy['auroc'] - m_b_full['auroc']:+.4f}, Cond={diag_acy['cond_number']:.1f})")


if __name__ == "__main__":
    main()
