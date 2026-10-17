"""
Training and evaluation pipeline for the separate GINE Validity Detector.

Training Policy:
- Train data: id_train (5,432 graphs) + corrupt_train (1,086 graphs, Families A & B only).
- Validation data: id_val (679 graphs) + corrupt_val (543 graphs, Families A & B only).
- Held-out: Family C exists ONLY in corrupt_test and is strictly EXCLUDED.
- Checkpoint selection criterion: Highest validation AUROC on Family A (the clean valence test).
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data import get_dataloader
from src.toxicity_model import GINEToxicityModel
from src.validity_model import (
    ValidityDataset,
    GINEValidityDetector,
    collate_validity_graphs,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_validity(
    detector: GINEValidityDetector,
    valid_split: str = "id_val",
    corrupt_split: str = "corrupt_val",
    data_dir: str = "processed_v3",
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Evaluate the validity detector on validation sets and break down by corruption family.

    Returns:
        auc_all: Overall validation AUROC (Valid vs. All Corrupt)
        auc_fam_a: AUROC for Valid vs. Family A (Gold standard: count invariant)
        auc_fam_b: AUROC for Valid vs. Family B (Degree shifting)
        metrics_dict: Dictionary containing summary numbers and sample counts
    """
    detector.eval()

    # Load master metadata table to map corrupted graphs to their corruption family
    df_master = pd.read_csv(Path(data_dir) / "tables/master_table.csv")
    corrupt_meta = df_master[df_master["set_name"] == corrupt_split].copy().reset_index(drop=True)

    # 1. Score valid graphs
    valid_loader = get_dataloader(valid_split, root_dir=data_dir, batch_size=64, shuffle=False)
    valid_scores = []
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            scores = detector.score_batch(batch)
            valid_scores.extend(scores.cpu().tolist())

    # 2. Score corrupted graphs
    corrupt_loader = get_dataloader(corrupt_split, root_dir=data_dir, batch_size=64, shuffle=False)
    corrupt_scores = []
    corrupt_uids = []
    with torch.no_grad():
        for batch in corrupt_loader:
            batch = batch.to(device)
            scores = detector.score_batch(batch)
            corrupt_scores.extend(scores.cpu().tolist())
            corrupt_uids.extend(batch.uid)

    # Align scores with family labels
    uid_to_score = dict(zip(corrupt_uids, corrupt_scores))
    corrupt_meta["score"] = corrupt_meta["uid"].map(uid_to_score)

    scores_fam_a = corrupt_meta[corrupt_meta["family"] == "A"]["score"].tolist()
    scores_fam_b = corrupt_meta[corrupt_meta["family"] == "B"]["score"].tolist()

    # Calculate AUROCs
    # Negative class = 0 (valid), Positive class = 1 (corrupt)
    y_true_all = [0.0] * len(valid_scores) + [1.0] * len(corrupt_scores)
    y_scores_all = valid_scores + corrupt_scores
    auc_all = float(roc_auc_score(y_true_all, y_scores_all))

    y_true_a = [0.0] * len(valid_scores) + [1.0] * len(scores_fam_a)
    y_scores_a = valid_scores + scores_fam_a
    auc_fam_a = float(roc_auc_score(y_true_a, y_scores_a))

    y_true_b = [0.0] * len(valid_scores) + [1.0] * len(scores_fam_b)
    y_scores_b = valid_scores + scores_fam_b
    auc_fam_b = float(roc_auc_score(y_true_b, y_scores_b))

    metrics_dict = {
        "n_valid": len(valid_scores),
        "n_corrupt_all": len(corrupt_scores),
        "n_corrupt_fam_a": len(scores_fam_a),
        "n_corrupt_fam_b": len(scores_fam_b),
        "auc_all": auc_all,
        "auc_fam_a": auc_fam_a,
        "auc_fam_b": auc_fam_b,
    }
    return auc_all, auc_fam_a, auc_fam_b, metrics_dict


def train_validity(
    data_dir: str = "processed_v3",
    toxicity_checkpoint: str = "checkpoints/toxicity_gine/best_model.pt",
    output_dir: str = "checkpoints/validity_gine",
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    device_str: str = "cpu",
    smoke_test: bool = False,
    max_train_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train separate validity detector using id_train and corrupt_train.
    """
    set_seed(seed)
    device = torch.device(device_str)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logs_dir = out_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base GINEToxicityModel architecture from '{toxicity_checkpoint}'...")
    tox_ckpt = torch.load(toxicity_checkpoint, map_location=device, weights_only=False)
    tox_cfg = tox_ckpt["config"]

    base_encoder = GINEToxicityModel(
        in_node_dim=tox_cfg["in_node_dim"],
        in_edge_dim=tox_cfg["in_edge_dim"],
        hidden_dim=tox_cfg["hidden_dim"],
        num_layers=tox_cfg["num_layers"],
        num_tasks=tox_cfg["num_tasks"],
        dropout=tox_cfg["dropout"],
        pooling=tox_cfg["pooling"],
    )
    base_encoder.load_state_dict(tox_ckpt["model_state_dict"])

    embed_dim = base_encoder.embed_dim
    detector = GINEValidityDetector(encoder=base_encoder, embed_dim=embed_dim).to(device)

    # Dataset: id_train (5,432) + corrupt_train (1,086)
    print("Preparing ValidityDataset (id_train + corrupt_train)...")
    train_dataset = ValidityDataset(valid_split="id_train", corrupt_split="corrupt_train", root_dir=data_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_validity_graphs,
    )

    # Class weighting: 5,432 valid vs 1,086 corrupted (ratio ~5.0)
    pos_weight = torch.tensor([5432.0 / 1086.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(detector.parameters(), lr=lr, weight_decay=weight_decay)

    config = {
        "model_architecture": "GINEValidityDetector",
        "encoder_base": toxicity_checkpoint,
        "embed_dim": embed_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "pos_weight": float(pos_weight.item()),
        "seed": seed,
        "selection_criterion": "highest validation Family A AUROC",
        "splits_used": {
            "train": "id_train (5,432 graphs) + corrupt_train (1,086 graphs; Families A & B)",
            "validation": "id_val (679 graphs) + corrupt_val (543 graphs; Families A & B)",
            "test": "UNTOUCHED (0 test graphs used)",
            "held_out_family_c": "STRICTLY UNSEEN (0 Family C graphs loaded)",
        },
    }
    with open(out_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    best_val_auc_a = -1.0
    best_epoch = -1
    best_checkpoint_path = out_path / "best_validity_model.pt"
    history = []

    print(f"\nStarting validity detector training for {epochs} epochs on {device}...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        detector.train()
        total_loss = 0.0
        n_batches = 0

        for b_idx, (batch_graphs, labels) in enumerate(train_loader):
            if max_train_batches and b_idx >= max_train_batches:
                break
            batch_graphs = batch_graphs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = detector(batch_graphs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation evaluation: breakdown by Family A and B
        auc_all, auc_a, auc_b, metrics = evaluate_validity(
            detector=detector,
            valid_split="id_val",
            corrupt_split="corrupt_val",
            data_dir=data_dir,
            device=device,
        )

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val AUROC All: {auc_all:.4f} | "
            f"Fam A: {auc_a:.4f} | "
            f"Fam B: {auc_b:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "auc_all": auc_all,
            "auc_fam_a": auc_a,
            "auc_fam_b": auc_b,
        })

        # Checkpoint selection strictly on Family A AUROC (the clean valence test)
        if auc_a > best_val_auc_a:
            best_val_auc_a = auc_a
            best_epoch = epoch
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Fam A AUROC: {auc_a:.4f})")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": detector.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auc_fam_a": auc_a,
                    "val_auc_fam_b": auc_b,
                    "val_auc_all": auc_all,
                    "config": config,
                },
                best_checkpoint_path,
            )

    total_time = time.time() - start_time
    print(f"\nValidity detector training completed in {total_time:.1f}s.")
    print(f"Best Val Family A AUROC: {best_val_auc_a:.4f} achieved at Epoch {best_epoch}.")

    with open(logs_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "best_epoch": best_epoch,
        "best_val_auc_fam_a": best_val_auc_a,
        "checkpoint_path": str(best_checkpoint_path),
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GINE Validity Detector.")
    parser.add_argument("--data_dir", type=str, default="processed_v3")
    parser.add_argument("--toxicity_checkpoint", type=str, default="checkpoints/toxicity_gine/best_model.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints/validity_gine")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true", help="Quick smoke test on truncated batches")

    args = parser.parse_args()

    if args.smoke_test:
        print("=== RUNNING VALIDITY TRAINING SMOKE TEST (2 Epochs, truncated) ===")
        train_validity(
            data_dir=args.data_dir,
            toxicity_checkpoint=args.toxicity_checkpoint,
            output_dir="checkpoints/smoke_validity_gine",
            epochs=2,
            batch_size=32,
            seed=args.seed,
            smoke_test=True,
            max_train_batches=5,
        )
    else:
        train_validity(
            data_dir=args.data_dir,
            toxicity_checkpoint=args.toxicity_checkpoint,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
