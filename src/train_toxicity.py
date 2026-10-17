"""
Multi-task GINE training and evaluation pipeline for Tox21 toxicity prediction.

Training Policy:
- Train set: strictly 'id_train' (5,432 graphs).
- Validation set: strictly 'id_val' (679 graphs) for checkpoint selection and metric tracking.
- Test sets: NEVER used during training or checkpoint selection.
- Objective: Masked multi-task BCE loss on 12 Tox21 endpoints.
- Checkpoint selection criterion: Best validation macro ROC-AUC.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.optim as optim

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data import get_dataloader
from src.toxicity_model import GINEToxicityModel
from src.loss import masked_multitask_bce_loss, compute_multitask_roc_auc


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float, Dict[int, float]]:
    """
    Evaluate model on a given dataloader.

    Returns:
        avg_loss: Average masked BCE loss
        mean_auc: Macro-averaged ROC-AUC across evaluatable tasks
        task_aucs: Per-task ROC-AUC scores
    """
    model.eval()
    total_loss = 0.0
    total_observed = 0
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits, _ = model(batch)
            loss, n_obs = masked_multitask_bce_loss(logits, batch.y)

            total_loss += loss.item() * n_obs
            total_observed += n_obs

            probs = torch.sigmoid(logits)
            y_true_list.append(batch.y.cpu().numpy())
            y_pred_list.append(probs.cpu().numpy())

    avg_loss = total_loss / max(total_observed, 1)
    y_true_all = np.concatenate(y_true_list, axis=0)
    y_pred_all = np.concatenate(y_pred_list, axis=0)

    mean_auc, task_aucs = compute_multitask_roc_auc(y_true_all, y_pred_all)
    return avg_loss, mean_auc, task_aucs


def train_toxicity(
    data_dir: str = "processed_v3",
    output_dir: str = "checkpoints/toxicity_gine",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dim: int = 128,
    num_layers: int = 4,
    dropout: float = 0.1,
    pooling: str = "mean_sum",
    seed: int = 42,
    device_str: str = "cpu",
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train GINEToxicityModel on id_train and validate on id_val.
    """
    set_seed(seed)
    device = torch.device(device_str)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logs_dir = out_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from '{data_dir}'...")
    train_loader = get_dataloader(
        split_name="id_train",
        root_dir=data_dir,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = get_dataloader(
        split_name="id_val",
        root_dir=data_dir,
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Building GINEToxicityModel (hidden_dim={hidden_dim}, layers={num_layers}, pooling={pooling})...")
    model = GINEToxicityModel(
        in_node_dim=30,
        in_edge_dim=6,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_tasks=12,
        dropout=dropout,
        pooling=pooling,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    config = {
        "model_architecture": "GINEToxicityModel",
        "in_node_dim": 30,
        "in_edge_dim": 6,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_tasks": 12,
        "dropout": dropout,
        "pooling": pooling,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "device": device_str,
        "epochs": epochs,
        "selection_criterion": "highest validation macro ROC-AUC",
        "splits_used": {
            "train": "id_train (5,432 graphs)",
            "validation": "id_val (679 graphs)",
            "test": "UNTOUCHED (0 test graphs used)",
        },
    }
    with open(out_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    best_val_auc = -1.0
    best_epoch = -1
    best_checkpoint_path = out_path / "best_model.pt"
    history = []

    print(f"\nStarting training for {epochs} epochs on {device}...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_obs = 0

        for b_idx, batch in enumerate(train_loader):
            if max_train_batches and b_idx >= max_train_batches:
                break
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss, n_obs = masked_multitask_bce_loss(logits, batch.y)

            if n_obs > 0:
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * n_obs
                train_obs += n_obs

        train_avg_loss = train_loss / max(train_obs, 1)

        # Validation evaluation
        val_loss, val_auc, task_aucs = evaluate(model, val_loader, device)
        scheduler.step(val_auc)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_avg_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro ROC-AUC: {val_auc:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_avg_loss,
            "val_loss": val_loss,
            "val_macro_auc": val_auc,
            "task_aucs": task_aucs,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val AUC: {val_auc:.4f})")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "config": config,
                },
                best_checkpoint_path,
            )

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s.")
    print(f"Best Val Macro ROC-AUC: {best_val_auc:.4f} achieved at Epoch {best_epoch}.")

    # Save training history
    with open(logs_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "checkpoint_path": str(best_checkpoint_path),
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GINE model for Tox21 toxicity prediction.")
    parser.add_argument("--data_dir", type=str, default="processed_v3", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/toxicity_gine", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="GINE layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke_test", action="store_true", help="Run 2 quick epochs for smoke testing")

    args = parser.parse_args()

    if args.smoke_test:
        print("=== RUNNING TRAINING SMOKE TEST (2 Epochs, truncated batches) ===")
        train_toxicity(
            data_dir=args.data_dir,
            output_dir="checkpoints/smoke_toxicity_gine",
            epochs=2,
            batch_size=32,
            hidden_dim=64,
            num_layers=2,
            seed=args.seed,
            max_train_batches=5,
            max_val_batches=3,
        )
    else:
        train_toxicity(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            seed=args.seed,
        )
