"""
Task C: Systematic Novelty Improvement Experiments.

Tests a justified set of alternative novelty estimators fitted strictly on id_train
and evaluated on validation sets (id_val, novel_scaffold_val, novel_dataset_val):
1. Size-conditioned / size-residualized covariance estimation.
2. Alternative pooling choices: Mean-only slice (128d), L2-normalized representations.
3. Latent neighborhood / non-parametric estimators: Cosine and Euclidean kNN distance.
4. Output-space Energy score (multi-task negative free energy).
5. Topological Weisfeiler-Lehman (WL) subtree histogram representation.

Reports AUROC, AUPR, FPR95, and 95% bootstrap confidence intervals for:
- Scaffold Novelty (Raw vs id_val)
- Scaffold Novelty (Matched Controls via matched_controls.csv)
- Dataset Novelty (Raw vs id_val)
- Dataset Novelty (Matched Controls via matched_controls.csv)
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
from scipy.linalg import pinvh
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data import get_dataloader
from src.evaluate import compute_fpr95, compute_ood_metrics
from src.toxicity_model import GINEToxicityModel


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval for AUROC."""
    rng = np.random.RandomState(seed)
    y_t = np.asarray(y_true)
    y_s = np.asarray(y_score)
    n = len(y_t)
    aucs: List[float] = []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_t[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y_t[idx], y_s[idx])))

    if len(aucs) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def run_novelty_experiments(
    data_dir: str = "processed_v3",
    output_dir: str = "step6_audit",
    batch_size: int = 128,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    print("=" * 70)
    print("TASK C: NOVELTY IMPROVEMENT EXPERIMENTS (VALIDATION ONLY)")
    print("=" * 70)

    # 1. Load backbone and extract representations
    print("\n[1/5] Extracting latent graph representations and logits from frozen backbone...")
    tox_ckpt = torch.load("checkpoints/toxicity_gine/best_model.pt", map_location=dev)
    cfg = tox_ckpt["config"]
    model = GINEToxicityModel(
        in_node_dim=cfg["in_node_dim"],
        in_edge_dim=cfg["in_edge_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_tasks=cfg["num_tasks"],
        dropout=cfg["dropout"],
        pooling=cfg["pooling"],
    )
    model.load_state_dict(tox_ckpt["model_state_dict"])
    model.to(dev)
    model.eval()

    def get_embs_logits_uids(split_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        loader = get_dataloader(split_name, root_dir=data_dir, batch_size=batch_size, shuffle=False)
        embs_list, logits_list, uids_list = [], [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(dev)
                embs = model.get_embeddings(b).cpu().numpy()
                logits, _ = model(b)
                embs_list.append(embs)
                logits_list.append(logits.cpu().numpy())
                uids_list.extend(b.uid)
        return (
            np.concatenate(embs_list, axis=0).astype(np.float32),
            np.concatenate(logits_list, axis=0).astype(np.float32),
            uids_list,
        )

    splits = ["id_train", "id_val", "novel_scaffold_val", "novel_dataset_val"]
    embs: Dict[str, np.ndarray] = {}
    logits: Dict[str, np.ndarray] = {}
    uids: Dict[str, List[str]] = {}

    for s in splits:
        e, l, u = get_embs_logits_uids(s)
        embs[s], logits[s], uids[s] = e, l, u
        print(f"  - Extracted {s:<20}: {len(u):>4} graphs (emb shape: {e.shape})")

    # Load metadata and matched controls
    master_df = pd.read_csv(Path(data_dir) / "tables/master_table.csv").set_index("uid")
    controls_df = pd.read_csv(Path(data_dir) / "tables/matched_controls.csv")

    sizes = {
        s: master_df.loc[uids[s], ["n_atoms", "n_bonds"]].to_numpy().astype(np.float32)
        for s in splits
    }
    atoms = {s: sizes[s][:, 0] for s in splits}

    scaff_matched_df = controls_df[(controls_df["target_set"] == "novel_scaffold_val") & (controls_df["matched"] == True)]
    data_matched_df = controls_df[(controls_df["target_set"] == "novel_dataset_val") & (controls_df["matched"] == True)]

    # ------------------------------------------------------------------
    # 2. Fit Methods strictly on id_train
    # ------------------------------------------------------------------
    print("\n[2/5] Fitting candidate novelty estimators strictly on id_train...")
    methods: Dict[str, Dict[str, np.ndarray]] = {}

    # Method 1: Baseline Mahalanobis (256d)
    lw_base = LedoitWolf().fit(embs["id_train"])
    mu_base = embs["id_train"].mean(axis=0)
    prec_base = pinvh(lw_base.covariance_ + 1e-6 * np.eye(256))

    def score_base_maha(E: np.ndarray) -> np.ndarray:
        diff = E - mu_base
        return np.sqrt(np.clip(np.sum((diff @ prec_base) * diff, axis=1), 0, None))

    methods["1. Baseline Mahalanobis (256d)"] = {s: score_base_maha(embs[s]) for s in splits}

    # Method 2: Mean-only slice (128d, size-invariant pooling)
    lw_mean = LedoitWolf().fit(embs["id_train"][:, :128])
    mu_mean = embs["id_train"][:, :128].mean(axis=0)
    prec_mean = pinvh(lw_mean.covariance_ + 1e-6 * np.eye(128))

    def score_mean_maha(E: np.ndarray) -> np.ndarray:
        diff = E[:, :128] - mu_mean
        return np.sqrt(np.clip(np.sum((diff @ prec_mean) * diff, axis=1), 0, None))

    methods["2. Mean-Only Slice (128d) Mahalanobis"] = {s: score_mean_maha(embs[s]) for s in splits}

    # Method 3: L2-Normalized 256d Mahalanobis
    norm_embs = {
        s: embs[s] / np.linalg.norm(embs[s], axis=1, keepdims=True) for s in splits
    }
    lw_norm = LedoitWolf().fit(norm_embs["id_train"])
    mu_norm = norm_embs["id_train"].mean(axis=0)
    prec_norm = pinvh(lw_norm.covariance_ + 1e-6 * np.eye(256))

    def score_norm_maha(NE: np.ndarray) -> np.ndarray:
        diff = NE - mu_norm
        return np.sqrt(np.clip(np.sum((diff @ prec_norm) * diff, axis=1), 0, None))

    methods["3. L2-Normalized (256d) Mahalanobis"] = {s: score_norm_maha(norm_embs[s]) for s in splits}

    # Method 4: Size-Residualized Embeddings Mahalanobis
    # Regress out size [n_atoms, n_bonds] linearly on id_train
    ridge = Ridge(alpha=1.0).fit(sizes["id_train"], embs["id_train"])
    res_embs = {s: embs[s] - ridge.predict(sizes[s]) for s in splits}

    lw_res = LedoitWolf().fit(res_embs["id_train"])
    mu_res = res_embs["id_train"].mean(axis=0)
    prec_res = pinvh(lw_res.covariance_ + 1e-6 * np.eye(256))

    def score_res_maha(RE: np.ndarray) -> np.ndarray:
        diff = RE - mu_res
        return np.sqrt(np.clip(np.sum((diff @ prec_res) * diff, axis=1), 0, None))

    methods["4. Size-Residualized Embedding Mahalanobis"] = {s: score_res_maha(res_embs[s]) for s in splits}

    # Method 5: Size-Residualized Distance
    # Regress Mahalanobis distance directly on [n_atoms, n_bonds] on id_train
    d_train = methods["1. Baseline Mahalanobis (256d)"]["id_train"]
    ridge_d = Ridge(alpha=1.0).fit(sizes["id_train"], d_train)

    methods["5. Size-Residualized Distance"] = {
        s: methods["1. Baseline Mahalanobis (256d)"][s] - ridge_d.predict(sizes[s])
        for s in splits
    }

    # Method 6: Size-Stratified Mahalanobis (4 quartile bins of id_train atoms)
    q_cuts = np.percentile(atoms["id_train"], [25, 50, 75])
    bin_models = {}
    for b in range(4):
        mask_b = (np.digitize(atoms["id_train"], q_cuts) == b)
        e_b = embs["id_train"][mask_b]
        lw_b = LedoitWolf().fit(e_b)
        bin_models[b] = (e_b.mean(axis=0), pinvh(lw_b.covariance_ + 1e-5 * np.eye(256)))

    def score_stratified(E: np.ndarray, A: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(E), dtype=np.float32)
        bins = np.digitize(A, q_cuts)
        for b in range(4):
            mb = (bins == b)
            if not np.any(mb):
                continue
            mu_b, prec_b = bin_models[b]
            diff = E[mb] - mu_b
            scores[mb] = np.sqrt(np.clip(np.sum((diff @ prec_b) * diff, axis=1), 0, None))
        return scores

    methods["6. Size-Stratified Mahalanobis (4 Bins)"] = {
        s: score_stratified(embs[s], atoms[s]) for s in splits
    }

    # Method 7: Latent Cosine kNN (k = 5, 10, 20)
    for k in [5, 10, 20]:
        knn_cos = NearestNeighbors(n_neighbors=k, metric="cosine").fit(norm_embs["id_train"])
        methods[f"7. Latent Cosine kNN (k={k})"] = {
            s: knn_cos.kneighbors(norm_embs[s])[0].mean(axis=1) for s in splits
        }

    # Method 8: Multi-Task Free Energy Score
    # Free energy for multi-task binary classifiers: higher = ID/confident, lower = OOD
    # Novelty score = -FreeEnergy = sum of softplus(logit) + softplus(-logit)
    def multi_task_energy(L: np.ndarray) -> np.ndarray:
        L_clip = np.clip(L, -20.0, 20.0)
        # log(1 + exp(L)) + log(1 + exp(-L))
        term = np.log1p(np.exp(L_clip)) + np.log1p(np.exp(-L_clip))
        return -np.sum(term, axis=1)

    methods["8. Multi-Task Free Energy Score"] = {s: multi_task_energy(logits[s]) for s in splits}

    # Method 9: Weisfeiler-Lehman (WL) Subtree Histogram Cosine kNN
    # 2-step WL subtree representation
    def compute_wl_counts(g_list: List[Dict[str, Any]]) -> List[Counter]:
        counts = []
        for g in g_list:
            edge_index = g["edge_index"].numpy()
            x = g["x"].numpy()
            num_nodes = x.shape[0]
            colors = np.argmax(x[:, :13], axis=1).tolist()
            adj = {i: [] for i in range(num_nodes)}
            for src, dst in zip(edge_index[0], edge_index[1]):
                adj[src].append(dst)
            all_c = list(colors)
            for _ in range(2):
                new_c = []
                for i in range(num_nodes):
                    tup = (colors[i], tuple(sorted([colors[j] for j in adj[i]])))
                    new_c.append(hash(tup))
                colors = new_c
                all_c.extend(colors)
            counts.append(Counter(all_c))
        return counts

    print("  - Extracting 2-WL subtree features for topological baseline...")
    raw_graphs = {s: torch.load(Path(data_dir) / f"graphs/{s}.pt", weights_only=True) for s in splits}
    wl_counts = {s: compute_wl_counts(raw_graphs[s]) for s in splits}

    train_vocab = Counter()
    for c in wl_counts["id_train"]:
        train_vocab.update(c)
    top_subtrees = [st for st, _ in train_vocab.most_common(500)]
    st_to_idx = {st: i for i, st in enumerate(top_subtrees)}

    def wl_to_mat(c_list: List[Counter]) -> np.ndarray:
        mat = np.zeros((len(c_list), len(top_subtrees)), dtype=np.float32)
        for r, c in enumerate(c_list):
            for st, cnt in c.items():
                if st in st_to_idx:
                    mat[r, st_to_idx[st]] = cnt
        return mat

    wl_mat = {s: wl_to_mat(wl_counts[s]) for s in splits}
    wl_norm = {
        s: wl_mat[s] / (np.linalg.norm(wl_mat[s], axis=1, keepdims=True) + 1e-8)
        for s in splits
    }
    knn_wl = NearestNeighbors(n_neighbors=10, metric="cosine").fit(wl_norm["id_train"])
    methods["9. WL Subtree Kernel Cosine kNN (k=10)"] = {
        s: knn_wl.kneighbors(wl_norm[s])[0].mean(axis=1) for s in splits
    }

    # ------------------------------------------------------------------
    # 3. Evaluate All Methods on Validation Benchmarks
    # ------------------------------------------------------------------
    print("\n[3/5] Evaluating all methods on validation sets with 95% Bootstrap CIs...")
    table_rows: List[Dict[str, Any]] = []

    for name, s_dict in methods.items():
        s_id = s_dict["id_val"]
        s_scaff = s_dict["novel_scaffold_val"]
        s_data = s_dict["novel_dataset_val"]

        # Scaffold Raw
        y_sr = np.array([0] * len(s_id) + [1] * len(s_scaff))
        scores_sr = np.concatenate([s_id, s_scaff])
        m_sr = compute_ood_metrics(y_sr, scores_sr)
        ci_sr = bootstrap_auroc_ci(y_sr, scores_sr, n_boot=500, seed=seed)

        # Scaffold Matched
        scaff_map = dict(zip(uids["novel_scaffold_val"], s_scaff))
        id_map = dict(zip(uids["id_val"], s_id))
        s_sm_ctrl = scaff_matched_df["control_uid"].map(id_map).to_numpy()
        s_sm_tgt = scaff_matched_df["target_uid"].map(scaff_map).to_numpy()
        y_sm = np.array([0] * len(scaff_matched_df) + [1] * len(scaff_matched_df))
        scores_sm = np.concatenate([s_sm_ctrl, s_sm_tgt])
        m_sm = compute_ood_metrics(y_sm, scores_sm)
        ci_sm = bootstrap_auroc_ci(y_sm, scores_sm, n_boot=500, seed=seed)

        # Dataset Raw
        y_dr = np.array([0] * len(s_id) + [1] * len(s_data))
        scores_dr = np.concatenate([s_id, s_data])
        m_dr = compute_ood_metrics(y_dr, scores_dr)
        ci_dr = bootstrap_auroc_ci(y_dr, scores_dr, n_boot=500, seed=seed)

        # Dataset Matched
        data_map = dict(zip(uids["novel_dataset_val"], s_data))
        s_dm_ctrl = data_matched_df["control_uid"].map(id_map).to_numpy()
        s_dm_tgt = data_matched_df["target_uid"].map(data_map).to_numpy()
        y_dm = np.array([0] * len(data_matched_df) + [1] * len(data_matched_df))
        scores_dm = np.concatenate([s_dm_ctrl, s_dm_tgt])
        m_dm = compute_ood_metrics(y_dm, scores_dm)
        ci_dm = bootstrap_auroc_ci(y_dm, scores_dm, n_boot=500, seed=seed)

        table_rows.append({
            "method": name,
            "scaff_raw_auroc": m_sr["auroc"],
            "scaff_raw_ci": ci_sr,
            "scaff_raw_aupr": m_sr["aupr"],
            "scaff_raw_fpr95": m_sr["fpr95"],
            "scaff_matched_auroc": m_sm["auroc"],
            "scaff_matched_ci": ci_sm,
            "scaff_matched_aupr": m_sm["aupr"],
            "scaff_matched_fpr95": m_sm["fpr95"],
            "dataset_raw_auroc": m_dr["auroc"],
            "dataset_raw_ci": ci_dr,
            "dataset_raw_aupr": m_dr["aupr"],
            "dataset_raw_fpr95": m_dr["fpr95"],
            "dataset_matched_auroc": m_dm["auroc"],
            "dataset_matched_ci": ci_dm,
            "dataset_matched_aupr": m_dm["aupr"],
            "dataset_matched_fpr95": m_dm["fpr95"],
        })

    # Format display table
    display_rows = []
    for r in table_rows:
        display_rows.append({
            "Method": r["method"],
            "Scaff Raw AUC [95% CI]": f"{r['scaff_raw_auroc']:.4f} [{r['scaff_raw_ci'][0]:.3f}-{r['scaff_raw_ci'][1]:.3f}]",
            "Scaff Match AUC [95% CI]": f"{r['scaff_matched_auroc']:.4f} [{r['scaff_matched_ci'][0]:.3f}-{r['scaff_matched_ci'][1]:.3f}]",
            "Data Raw AUC [95% CI]": f"{r['dataset_raw_auroc']:.4f} [{r['dataset_raw_ci'][0]:.3f}-{r['dataset_raw_ci'][1]:.3f}]",
            "Data Match AUC [95% CI]": f"{r['dataset_matched_auroc']:.4f} [{r['dataset_matched_ci'][0]:.3f}-{r['dataset_matched_ci'][1]:.3f}]",
        })

    df_display = pd.DataFrame(display_rows)
    print("\n" + "=" * 120)
    print("NOVELTY EXPERIMENT VALIDATION COMPARISON (AUROC with 95% Bootstrap Confidence Intervals)")
    print("=" * 120)
    print(df_display.to_string(index=False))
    print("=" * 120)

    # ------------------------------------------------------------------
    # 4. Save Deliverables
    # ------------------------------------------------------------------
    out_json = out_path / "novelty_experiment_results.json"
    with open(out_json, "w") as f:
        json.dump(table_rows, f, indent=2)

    df_full = pd.DataFrame(table_rows)
    df_full.to_csv(out_path / "novelty_experiment_results.csv", index=False)
    print(f"\n[4/5] Saved detailed experiment results to {out_json} and CSV.")

    return {"results": table_rows}


if __name__ == "__main__":
    run_novelty_experiments()
