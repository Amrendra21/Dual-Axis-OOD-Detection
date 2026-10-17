"""
Fitting script for the Mahalanobis Novelty Estimator.

Workflow
--------
1. Load the trained toxicity GNN backbone from checkpoint.
2. Extract graph embeddings for `id_train` (the only permitted fitting split).
3. Fit the Ledoit-Wolf regularized Mahalanobis estimator on those embeddings.
4. Save the fitted estimator and a reproducible config JSON.
5. Evaluate novelty scores on `id_val`, `novel_scaffold_val`, and `novel_dataset_val`
   and report summary statistics (no test-split evaluation performed here).

Usage
-----
    python src/fit_novelty.py \\
        --checkpoint checkpoints/toxicity_gine/best_model.pt \\
        --output_dir checkpoints/novelty_mahal \\
        --batch_size 256

The fit and evaluation results are written to:
    checkpoints/novelty_mahal/estimator.npz      -- Serialized estimator
    checkpoints/novelty_mahal/config.json        -- Reproducible configuration
    checkpoints/novelty_mahal/val_scores.npz     -- Validation split scores + UIDs
    checkpoints/novelty_mahal/fit_report.json    -- Summary statistics

Strict holdout policy
---------------------
- `id_train` only is used for fitting. No other split is used during fitting.
- `id_test`, `novel_scaffold_test`, `novel_dataset_test`, and all `corrupt_*` test
  splits are never loaded in this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.data import MolecularDataset
from src.toxicity_model import GINEToxicityModel
from src.novelty_model import (
    MahalanobisNoveltyEstimator,
    extract_embeddings,
    PERMITTED_FIT_SPLITS,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s:%(lineno)d  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fit_novelty")


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_toxicity_encoder(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[GINEToxicityModel, Dict]:
    """
    Load the trained toxicity GNN backbone from a checkpoint.

    Args:
        checkpoint_path: Path to `best_model.pt`.
        device:          Device string.

    Returns:
        model:  GINEToxicityModel in eval mode with weights loaded.
        config: Configuration dictionary stored in the checkpoint.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    config = ckpt.get("config", {})

    model = GINEToxicityModel(
        in_node_dim=config.get("in_node_dim", 30),
        in_edge_dim=config.get("in_edge_dim", 6),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 4),
        num_tasks=config.get("num_tasks", 12),
        dropout=config.get("dropout", 0.1),
        pooling=config.get("pooling", "mean_sum"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(
        "Loaded toxicity GNN from '%s'  (epoch=%s, val_auc=%s, embed_dim=%d).",
        ckpt_path.name,
        ckpt.get("epoch", "?"),
        f"{ckpt.get('val_auc', float('nan')):.4f}",
        model.embed_dim,
    )
    return model, config


# ---------------------------------------------------------------------------
# Summary statistics helper
# ---------------------------------------------------------------------------

def score_statistics(scores: np.ndarray, split_name: str) -> Dict:
    """Compute and log summary statistics for a score array."""
    stats = {
        "split": split_name,
        "n": int(scores.shape[0]),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "p95": float(np.percentile(scores, 95)),
        "max": float(np.max(scores)),
    }
    logger.info(
        "  %-30s  n=%5d  mean=%.2f  std=%.2f  median=%.2f  [min=%.2f, max=%.2f]",
        split_name,
        stats["n"], stats["mean"], stats["std"], stats["median"],
        stats["min"], stats["max"],
    )
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    logger.info("=" * 65)
    logger.info("STEP 4: Mahalanobis Novelty Estimator — Fit & Evaluate")
    logger.info("=" * 65)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    # -------------------------------------------------------------------
    # 1. Load encoder
    # -------------------------------------------------------------------
    logger.info("[1/5] Loading toxicity GNN backbone from checkpoint...")
    encoder, ckpt_config = load_toxicity_encoder(args.checkpoint, device=args.device)

    # Verify embed_dim
    expected_embed_dim = encoder.embed_dim
    logger.info("Encoder embed_dim = %d", expected_embed_dim)
    assert expected_embed_dim == 256, (
        f"Expected 256-dim embeddings (hidden_dim=128, pooling='mean_sum'), "
        f"but got embed_dim={expected_embed_dim}. Check the checkpoint config."
    )

    # -------------------------------------------------------------------
    # 2. Extract id_train embeddings
    # -------------------------------------------------------------------
    logger.info("[2/5] Extracting id_train embeddings (fitting set only)...")
    train_embeddings, train_uids = extract_embeddings(
        encoder=encoder,
        split_name="id_train",
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
    logger.info(
        "id_train embeddings: shape=%s, dtype=%s, finite=%s",
        train_embeddings.shape,
        train_embeddings.dtype,
        bool(np.all(np.isfinite(train_embeddings))),
    )

    # -------------------------------------------------------------------
    # 3. Fit the estimator
    # -------------------------------------------------------------------
    logger.info("[3/5] Fitting Ledoit-Wolf regularized Mahalanobis estimator...")
    estimator = MahalanobisNoveltyEstimator()
    estimator.fit(train_embeddings, fit_split="id_train")

    logger.info(
        "Estimator fitted: shrinkage=%.6f, embed_dim=%d, n_train=%d",
        estimator.shrinkage,
        estimator.embed_dim,
        estimator.n_train,
    )

    # Self-check: score the training set and verify finite output
    train_scores = estimator.score(train_embeddings)
    assert np.all(np.isfinite(train_scores)), (
        "Training set novelty scores contain NaN/Inf — check the precision matrix."
    )
    logger.info("Self-check passed: all id_train scores are finite.")

    # -------------------------------------------------------------------
    # 4. Save estimator
    # -------------------------------------------------------------------
    logger.info("[4/5] Saving estimator and configuration...")
    estimator_path = output_dir / "estimator.npz"
    estimator.save(estimator_path)

    # Verify round-trip serialization
    loaded_estimator = MahalanobisNoveltyEstimator.load(estimator_path)
    reloaded_scores = loaded_estimator.score(train_embeddings[:32])
    original_scores = estimator.score(train_embeddings[:32])
    assert np.allclose(reloaded_scores, original_scores, rtol=1e-5, atol=1e-6), (
        "Serialization round-trip check FAILED: reloaded scores differ from original!"
    )
    logger.info("Round-trip serialization check PASSED.")

    # -------------------------------------------------------------------
    # 5. Evaluate on validation splits (NO test splits)
    # -------------------------------------------------------------------
    logger.info("[5/5] Evaluating novelty scores on validation splits...")

    val_splits = [
        "id_val",
        "novel_scaffold_val",
        "novel_dataset_val",
    ]

    all_split_stats = []
    saved_scores = {}

    # Include training set stats for reference (already computed)
    train_stats = score_statistics(train_scores, "id_train (reference / fitting set)")
    all_split_stats.append(train_stats)
    saved_scores["id_train"] = {
        "uids": train_uids,
        "scores": train_scores.tolist(),
    }

    for split_name in val_splits:
        try:
            embs, uids = extract_embeddings(
                encoder=encoder,
                split_name=split_name,
                root_dir=args.root_dir,
                batch_size=args.batch_size,
                device=args.device,
            )
            scores = estimator.score(embs)
            assert np.all(np.isfinite(scores)), (
                f"Non-finite scores found for split '{split_name}'."
            )
            stats = score_statistics(scores, split_name)
            all_split_stats.append(stats)
            saved_scores[split_name] = {
                "uids": uids,
                "scores": scores.tolist(),
            }
        except FileNotFoundError as e:
            logger.warning("Split '%s' not found, skipping: %s", split_name, e)

    # -------------------------------------------------------------------
    # Save val scores and report
    # -------------------------------------------------------------------
    val_scores_path = output_dir / "val_scores.json"
    with open(val_scores_path, "w") as f:
        json.dump(saved_scores, f)
    logger.info("Validation scores saved to '%s'.", val_scores_path)

    t_elapsed = time.perf_counter() - t_start

    config_out = {
        "step": "Step 4 — Mahalanobis Novelty Estimator",
        "estimator": "MahalanobisNoveltyEstimator",
        "covariance_regularization": "Ledoit-Wolf shrinkage",
        "fit_split": "id_train",
        "n_train": estimator.n_train,
        "embed_dim": estimator.embed_dim,
        "shrinkage_coefficient": estimator.shrinkage,
        "score_direction": "higher = more novel / farther from ID distribution",
        "encoder_checkpoint": str(args.checkpoint),
        "encoder_embed_dim": int(expected_embed_dim),
        "encoder_val_auc": ckpt_config.get("val_auc", None),
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
        "runtime_seconds": round(t_elapsed, 2),
        "splits_used_for_fitting": list(PERMITTED_FIT_SPLITS),
        "splits_evaluated": [s["split"] for s in all_split_stats],
        "test_splits_used": [],
        "outputs": {
            "estimator": str(estimator_path),
            "val_scores": str(val_scores_path),
        },
        "strict_holdout_policy": (
            "id_test, novel_scaffold_test, novel_dataset_test, corrupt_test "
            "were NOT loaded in this script."
        ),
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_out, f, indent=2)
    logger.info("Configuration saved to '%s'.", config_path)

    fit_report = {
        "config": config_out,
        "split_statistics": all_split_stats,
    }
    report_path = output_dir / "fit_report.json"
    with open(report_path, "w") as f:
        json.dump(fit_report, f, indent=2)
    logger.info("Fit report saved to '%s'.", report_path)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("NOVELTY ESTIMATOR FIT COMPLETE")
    logger.info("=" * 65)
    logger.info("Elapsed time     : %.1f seconds", t_elapsed)
    logger.info("Shrinkage coeff  : %.6f", estimator.shrinkage)
    logger.info("Embed dim        : %d", estimator.embed_dim)
    logger.info("n_train          : %d", estimator.n_train)
    logger.info("")
    logger.info("Split novelty score statistics:")
    header = f"  {'Split':<32} {'N':>6}  {'Mean':>8}  {'Std':>8}  {'Median':>8}"
    logger.info(header)
    logger.info("  " + "-" * 70)
    for s in all_split_stats:
        logger.info(
            "  %-32s  %6d  %8.2f  %8.2f  %8.2f",
            s["split"], s["n"], s["mean"], s["std"], s["median"],
        )
    logger.info("=" * 65)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate the Mahalanobis novelty estimator (Step 4)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/toxicity_gine/best_model.pt",
        help="Path to the trained toxicity GNN checkpoint (default: checkpoints/toxicity_gine/best_model.pt)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="processed_v3",
        help="Path to processed_v3 directory or repo root (default: processed_v3)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/novelty_mahal",
        help="Directory for saved estimator, scores, and reports (default: checkpoints/novelty_mahal)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for embedding extraction (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for embedding extraction: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set seed for reproducibility (affects any numpy operations)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
