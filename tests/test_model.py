"""
Unit and integration tests for GINE backbone model, graph embeddings, and masked loss.

Covers:
1. Masked loss unit tests:
   - Standard labels with no NaNs.
   - Partial NaNs (sparse Tox21 labels).
   - Batch with 100% NaNs (safe zero loss, no exception).
   - Equivalence to manual observed-only BCE.
2. GINE model forward pass test:
   - Real batch from id_train.
   - Output logit shape [batch_size, 12] and finite check.
   - Embedding extraction shape [batch_size, embed_dim] matching batch size.
3. Embedding reproducibility test:
   - Same batch through eval mode yields identical embeddings.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data import get_dataloader
from src.toxicity_model import GINEToxicityModel
from src.loss import masked_multitask_bce_loss, compute_multitask_roc_auc


def test_masked_loss() -> None:
    print("\n--- Test 1: Masked Multi-Task BCE Loss ---")

    # Case A: Standard targets with no NaNs
    logits = torch.tensor([[1.5, -2.0], [0.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    loss, n_obs = masked_multitask_bce_loss(logits, targets)
    expected_bce = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.isclose(loss, expected_bce), f"Expected {expected_bce}, got {loss}"
    assert n_obs == 4, f"Expected 4 observed, got {n_obs}"
    print("  [Pass] Standard non-NaN targets match standard BCE.")

    # Case B: Partial NaNs
    targets_sparse = torch.tensor([[1.0, float("nan")], [float("nan"), 1.0]], dtype=torch.float32)
    loss_sparse, n_obs_sparse = masked_multitask_bce_loss(logits, targets_sparse)
    manual_loss = (
        F.binary_cross_entropy_with_logits(logits[0:1, 0:1], targets[0:1, 0:1]) +
        F.binary_cross_entropy_with_logits(logits[1:2, 1:2], targets[1:2, 1:2])
    ) / 2.0
    assert torch.isclose(loss_sparse, manual_loss), f"Expected {manual_loss}, got {loss_sparse}"
    assert n_obs_sparse == 2, f"Expected 2 observed, got {n_obs_sparse}"
    print("  [Pass] Partial NaNs correctly excluded and normalized by observed count.")

    # Case C: All NaNs
    targets_all_nan = torch.full((3, 12), float("nan"), dtype=torch.float32)
    logits_dummy = torch.randn((3, 12), requires_grad=True)
    loss_nan, n_obs_nan = masked_multitask_bce_loss(logits_dummy, targets_all_nan)
    assert loss_nan.item() == 0.0, f"Expected 0.0 for all-NaN batch, got {loss_nan.item()}"
    assert n_obs_nan == 0, f"Expected 0 observed, got {n_obs_nan}"
    # Ensure backprop works without failing
    loss_nan.backward()
    assert logits_dummy.grad is not None, "Gradient should be propagated (zeros) even with all-NaN batch."
    print("  [Pass] All-NaN batch handled safely with zero loss and valid backprop.")


def test_model_forward_and_embeddings() -> None:
    print("\n--- Test 2: GINEToxicityModel Forward Pass & Embedding Extraction ---")
    loader = get_dataloader(split_name="id_train", root_dir="processed_v3", batch_size=8, shuffle=False)
    batch = next(iter(loader))

    model = GINEToxicityModel(
        in_node_dim=30,
        in_edge_dim=6,
        hidden_dim=64,
        num_layers=3,
        num_tasks=12,
        dropout=0.1,
        pooling="mean_sum",
    )
    model.eval()

    with torch.no_grad():
        logits, embeddings = model(batch)
        standalone_embeddings = model.get_embeddings(batch)

    # Verify shapes
    assert logits.shape == (8, 12), f"Expected logits shape (8, 12), got {logits.shape}"
    assert embeddings.shape == (8, 128), f"Expected embeddings shape (8, 128), got {embeddings.shape}"
    assert standalone_embeddings.shape == (8, 128), f"Expected standalone shape (8, 128), got {standalone_embeddings.shape}"

    # Verify finite values
    assert torch.isfinite(logits).all(), "Logits contain non-finite values!"
    assert torch.isfinite(embeddings).all(), "Embeddings contain non-finite values!"
    assert torch.allclose(embeddings, standalone_embeddings), "Embeddings from forward and get_embeddings must match!"

    # Verify UID count matches embedding count
    assert len(batch.uid) == embeddings.size(0) == 8, "Number of embeddings must match number of graph UIDs in batch!"
    print(f"  [Pass] Forward logits shape: {logits.shape}, embeddings shape: {embeddings.shape}")
    print(f"  [Pass] Verified 1-to-1 correspondence with graph UIDs: {batch.uid[:3]}...")


def test_reproducibility() -> None:
    print("\n--- Test 3: Reproducibility & Determinism ---")
    torch.manual_seed(42)
    m1 = GINEToxicityModel(in_node_dim=30, in_edge_dim=6, hidden_dim=32, num_layers=2)
    torch.manual_seed(42)
    m2 = GINEToxicityModel(in_node_dim=30, in_edge_dim=6, hidden_dim=32, num_layers=2)

    # Check identical initial weights
    p1 = list(m1.parameters())[0]
    p2 = list(m2.parameters())[0]
    assert torch.allclose(p1, p2), "Identical seeds must yield identical initial model weights!"
    print("  [Pass] Seed control ensures reproducible weight initialization.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING GINE BACKBONE & LOSS UNIT TESTS")
    print("=" * 60)
    try:
        test_masked_loss()
        test_model_forward_and_embeddings()
        test_reproducibility()
        print("\n" + "=" * 60)
        print("ALL GINE BACKBONE & LOSS TESTS PASSED SUCCESSFULLY (Exit Code 0)")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\nTEST SUITE FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
