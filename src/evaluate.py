"""
Unified evaluation module for Dual-Axis OOD Detection.

Integrates the three frozen components:
1. GINEToxicityModel (toxicity backbone and embedding extractor)
2. GINEValidityDetector (structural validity / corruption classifier)
3. MahalanobisNoveltyEstimator (regularized density-based novelty scorer)

Provides metric computation (AUROC, AUPR, FPR95), split scoring,
and standardized prediction DataFrame export matching the agreed schema:
- uid
- benchmark/split
- true_label
- validity_score
- novelty_score
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import torch
import torch.nn as nn

from src.data import MolecularDataset, get_dataloader
from src.novelty_model import MahalanobisNoveltyEstimator
from src.toxicity_model import GINEToxicityModel
from src.validity_model import GINEValidityDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_fpr95(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute False Positive Rate at 95% True Positive Rate (FPR95).

    Convention:
        y_true = 0: In-Distribution / Negative class (e.g., valid or ID)
        y_true = 1: Out-of-Distribution / Positive class (e.g., corrupted or novel)
        y_score: Higher score indicates higher likelihood of positive (OOD) class.

    Returns:
        float: FPR (fraction of negative examples misclassified as positive)
               at the threshold achieving at least 95% TPR.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Edge cases
    if len(np.unique(y_true)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Find the smallest FPR where TPR >= 0.95
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def compute_ood_metrics(
    y_true: Union[List[int], np.ndarray],
    y_score: Union[List[float], np.ndarray],
) -> Dict[str, float]:
    """
    Compute standard OOD detection metrics:
    - AUROC (Area under Receiver Operating Characteristic curve)
    - AUPR (Area under Precision-Recall curve / Average Precision)
    - FPR95 (False Positive Rate at 95% True Positive Rate)

    Args:
        y_true: Binary ground truth labels (0 = negative/ID, 1 = positive/OOD).
        y_score: Continuous prediction scores (higher = more likely positive/OOD).

    Returns:
        dict with keys 'auroc', 'aupr', 'fpr95', 'n_neg', 'n_pos'.
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    y_s = np.asarray(y_score, dtype=np.float64)

    n_neg = int(np.sum(y_t == 0))
    n_pos = int(np.sum(y_t == 1))

    if n_neg == 0 or n_pos == 0:
        return {
            "auroc": float("nan"),
            "aupr": float("nan"),
            "fpr95": float("nan"),
            "n_neg": n_neg,
            "n_pos": n_pos,
        }

    auroc = float(roc_auc_score(y_t, y_s))
    aupr = float(average_precision_score(y_t, y_s))
    fpr95 = compute_fpr95(y_t, y_s)

    return {
        "auroc": auroc,
        "aupr": aupr,
        "fpr95": fpr95,
        "n_neg": n_neg,
        "n_pos": n_pos,
    }


# ---------------------------------------------------------------------------
# Component Loading
# ---------------------------------------------------------------------------

def load_frozen_components(
    toxicity_ckpt_path: Union[str, Path] = "checkpoints/toxicity_gine/best_model.pt",
    validity_ckpt_path: Union[str, Path] = "checkpoints/validity_gine/best_validity_model.pt",
    novelty_estimator_path: Union[str, Path] = "checkpoints/novelty_mahal/estimator.npz",
    device: Union[str, torch.device] = "cpu",
) -> Tuple[GINEToxicityModel, GINEValidityDetector, MahalanobisNoveltyEstimator]:
    """
    Load and verify all three frozen components.

    Checks:
    - Checkpoint files exist and load without error.
    - Architecture hyperparameters match saved configuration.
    - Encoder, validity detector, and novelty estimator share the identical
      embedding dimension (D = 256).
    - Models are set to eval() mode with gradients disabled.

    Returns:
        (toxicity_model, validity_detector, novelty_estimator)
    """
    device = torch.device(device)

    # 1. Load Toxicity Backbone
    tox_path = Path(toxicity_ckpt_path)
    if not tox_path.exists():
        raise FileNotFoundError(f"Toxicity checkpoint not found: {tox_path}")

    tox_ckpt = torch.load(tox_path, map_location=device)
    cfg = tox_ckpt.get("config", {})
    toxicity_model = GINEToxicityModel(
        in_node_dim=cfg.get("in_node_dim", 30),
        in_edge_dim=cfg.get("in_edge_dim", 6),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 4),
        num_tasks=cfg.get("num_tasks", 12),
        dropout=cfg.get("dropout", 0.1),
        pooling=cfg.get("pooling", "mean_sum"),
    )
    toxicity_model.load_state_dict(tox_ckpt["model_state_dict"])
    toxicity_model.to(device)
    toxicity_model.eval()

    # 2. Load Validity Detector
    val_path = Path(validity_ckpt_path)
    if not val_path.exists():
        raise FileNotFoundError(f"Validity checkpoint not found: {val_path}")

    val_ckpt = torch.load(val_path, map_location=device)
    val_enc = GINEToxicityModel(
        in_node_dim=cfg.get("in_node_dim", 30),
        in_edge_dim=cfg.get("in_edge_dim", 6),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 4),
        num_tasks=cfg.get("num_tasks", 12),
        dropout=cfg.get("dropout", 0.1),
        pooling=cfg.get("pooling", "mean_sum"),
    )
    validity_detector = GINEValidityDetector(encoder=val_enc, embed_dim=val_enc.embed_dim)
    validity_detector.load_state_dict(val_ckpt["model_state_dict"])
    validity_detector.to(device)
    validity_detector.eval()

    # 3. Load Mahalanobis Novelty Estimator
    nov_path = Path(novelty_estimator_path)
    if not nov_path.exists():
        raise FileNotFoundError(f"Novelty estimator artifact not found: {nov_path}")

    novelty_estimator = MahalanobisNoveltyEstimator.load(nov_path)

    # 4. Consistency checks
    if toxicity_model.embed_dim != validity_detector.embed_dim:
        raise ValueError(
            f"Embedding dimension mismatch: toxicity_model ({toxicity_model.embed_dim}) "
            f"!= validity_detector ({validity_detector.embed_dim})"
        )
    if toxicity_model.embed_dim != novelty_estimator.embed_dim:
        raise ValueError(
            f"Embedding dimension mismatch: toxicity_model ({toxicity_model.embed_dim}) "
            f"!= novelty_estimator ({novelty_estimator.embed_dim})"
        )

    logger.info(
        "Loaded all frozen components. Embed dim = %d, device = %s",
        toxicity_model.embed_dim,
        device,
    )
    return toxicity_model, validity_detector, novelty_estimator


# ---------------------------------------------------------------------------
# Split Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_split(
    split_name: str,
    toxicity_model: GINEToxicityModel,
    validity_detector: GINEValidityDetector,
    novelty_estimator: MahalanobisNoveltyEstimator,
    root_dir: Union[str, Path] = "processed_v3",
    batch_size: int = 64,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    Score all molecular graphs in a given split with both validity and novelty models,
    as well as toxicity multi-task predictions.

    Guarantees:
    - Strictly preserves graph UID order.
    - Finiteness check for all validity and novelty scores.
    - Toxicity predictions preserve raw labels and NaNs.

    Returns:
        dict containing:
        - 'uids': List[str]
        - 'validity_scores': np.ndarray of shape [N] (probabilities in [0, 1])
        - 'novelty_scores': np.ndarray of shape [N] (Mahalanobis distances >= 0)
        - 'toxicity_probs': np.ndarray of shape [N, 12] (probabilities in [0, 1])
        - 'toxicity_labels': np.ndarray of shape [N, 12] (ground truth, NaNs preserved)
    """
    device = torch.device(device)
    loader = get_dataloader(split_name, root_dir=root_dir, batch_size=batch_size, shuffle=False)

    uids: List[str] = []
    val_scores_list: List[np.ndarray] = []
    nov_scores_list: List[np.ndarray] = []
    tox_probs_list: List[np.ndarray] = []
    tox_labels_list: List[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        uids.extend(batch.uid)

        # 1. Validity score: sigmoid(logits) -> [0, 1]
        v_scores = validity_detector.score_batch(batch).cpu().numpy()
        val_scores_list.append(v_scores)

        # 2. Novelty score: Mahalanobis distance from ID train centroid
        embs = toxicity_model.get_embeddings(batch).cpu().numpy()
        n_scores = novelty_estimator.score(embs)
        nov_scores_list.append(n_scores)

        # 3. Toxicity multi-task predictions
        tox_logits, _ = toxicity_model(batch)
        tox_probs = torch.sigmoid(tox_logits).cpu().numpy()
        tox_probs_list.append(tox_probs)
        tox_labels_list.append(batch.y.cpu().numpy())

    validity_scores = np.concatenate(val_scores_list, axis=0).astype(np.float32)
    novelty_scores = np.concatenate(nov_scores_list, axis=0).astype(np.float32)
    toxicity_probs = np.concatenate(tox_probs_list, axis=0).astype(np.float32)
    toxicity_labels = np.concatenate(tox_labels_list, axis=0).astype(np.float32)

    # Sanity checks
    if not np.all(np.isfinite(validity_scores)):
        raise ValueError(f"Non-finite validity scores detected in split '{split_name}'.")
    if not np.all(np.isfinite(novelty_scores)):
        raise ValueError(f"Non-finite novelty scores detected in split '{split_name}'.")
    if len(uids) != len(validity_scores):
        raise ValueError(f"Length mismatch: {len(uids)} UIDs vs {len(validity_scores)} scores.")

    return {
        "uids": uids,
        "validity_scores": validity_scores,
        "novelty_scores": novelty_scores,
        "toxicity_probs": toxicity_probs,
        "toxicity_labels": toxicity_labels,
    }


# ---------------------------------------------------------------------------
# Prediction DataFrame Export
# ---------------------------------------------------------------------------

def build_benchmark_predictions_df(
    benchmark_tag: str,
    id_uids: List[str],
    id_val_scores: np.ndarray,
    id_nov_scores: np.ndarray,
    ood_uids: List[str],
    ood_val_scores: np.ndarray,
    ood_nov_scores: np.ndarray,
    id_split_label: str,
    ood_split_label: str,
) -> pd.DataFrame:
    """
    Construct a standardized predictions DataFrame for a single benchmark comparison.

    Schema:
    - uid: Unique molecular graph identifier
    - benchmark/split: e.g. 'validity/id_val', 'validity/corrupt_val'
    - true_label: 0 for ID/valid, 1 for OOD/corrupted
    - validity_score: Score from GINEValidityDetector in [0, 1] (higher = more corrupted)
    - novelty_score: Score from MahalanobisNoveltyEstimator (higher = more novel)

    Label Semantics:
    - For validity benchmarks: true_label indicates valid (0) vs corrupted (1).
    - For novelty benchmarks: true_label indicates ID (0) vs OOD/novel (1).
    """
    n_id = len(id_uids)
    n_ood = len(ood_uids)

    df_id = pd.DataFrame({
        "uid": id_uids,
        "benchmark/split": f"{benchmark_tag}/{id_split_label}",
        "true_label": [0] * n_id,
        "validity_score": id_val_scores,
        "novelty_score": id_nov_scores,
    })

    df_ood = pd.DataFrame({
        "uid": ood_uids,
        "benchmark/split": f"{benchmark_tag}/{ood_split_label}",
        "true_label": [1] * n_ood,
        "validity_score": ood_val_scores,
        "novelty_score": ood_nov_scores,
    })

    return pd.concat([df_id, df_ood], ignore_index=True)
