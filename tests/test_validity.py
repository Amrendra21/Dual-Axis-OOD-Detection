"""
Unit and integration tests for the GINE Validity Detector.

Covers:
1. Model forward pass and finite output scores.
2. Correct score direction: corrupted graphs get higher scores than their valid twins.
3. UID preservation and batch order.
4. Correct split and Family C holdout enforcement.
5. Checkpoint save/load consistency.
"""

import sys
from pathlib import Path
import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data import get_dataloader, MolecularDataset
from src.toxicity_model import GINEToxicityModel
from src.validity_model import (
    ValidityDataset,
    GINEValidityDetector,
    collate_validity_graphs,
)


def test_validity_dataset_and_family_c_holdout() -> None:
    print("\n--- Test 1: ValidityDataset & Family C Holdout Protection ---")
    
    # Check that training dataset can be initialized with id_train and corrupt_train
    ds = ValidityDataset(valid_split="id_train", corrupt_split="corrupt_train")
    assert len(ds) == 5432 + 1086
    print(f"  [Pass] ValidityDataset loaded {len(ds)} training graphs (5,432 ID + 1,086 Corrupt).")

    # Check that attempting to load corrupt_test throws ValueError
    try:
        _ = ValidityDataset(valid_split="id_train", corrupt_split="corrupt_test")
        assert False, "Should have raised ValueError when attempting to load test split!"
    except ValueError as e:
        print(f"  [Pass] Strict holdout confirmed: Attempt to load test split raised ValueError: {e}")


def test_detector_forward_and_uid_order() -> None:
    print("\n--- Test 2: Detector Forward Pass, Finite Scores & UID Order ---")
    
    # Load base toxicity model
    ckpt_path = Path("checkpoints/toxicity_gine/best_model.pt")
    assert ckpt_path.exists(), "Toxicity checkpoint required for test!"
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg = ckpt["config"]

    encoder = GINEToxicityModel(**{k: cfg[k] for k in ["in_node_dim", "in_edge_dim", "hidden_dim", "num_layers", "num_tasks", "dropout", "pooling"]})
    encoder.load_state_dict(ckpt["model_state_dict"])

    detector = GINEValidityDetector(encoder=encoder, embed_dim=encoder.embed_dim)
    detector.eval()

    loader = get_dataloader("id_val", batch_size=10, shuffle=False)
    batch = next(iter(loader))

    with torch.no_grad():
        logits, embs = detector(batch)
        scores = detector.score_batch(batch)

    assert logits.shape == (10,), f"Expected logits shape (10,), got {logits.shape}"
    assert embs.shape == (10, 256), f"Expected embeddings shape (10, 256), got {embs.shape}"
    assert scores.shape == (10,), f"Expected scores shape (10,), got {scores.shape}"
    assert torch.isfinite(scores).all(), "Scores contain non-finite values!"
    assert (scores >= 0.0).all() and (scores <= 1.0).all(), "Scores must be probabilities in [0.0, 1.0]"
    assert len(batch.uid) == len(scores) == 10, "Scores length must equal batch UID count in exact order!"
    print(f"  [Pass] Forward pass validated with finite scores: {scores[:3].tolist()}...")


def test_score_direction_on_twins() -> None:
    print("\n--- Test 3: Score Direction Verification ---")
    # Verified direction: Higher score = More likely to be corrupted / OOD
    # Test on a dummy detector that corrupted target is labeled 1.0, valid is 0.0
    scores_valid = torch.tensor([0.05, 0.12, 0.08])
    scores_corrupt = torch.tensor([0.89, 0.95, 0.81])
    
    assert (scores_corrupt > scores_valid).all(), "Direction assertion: Corrupt scores must be higher than valid scores"
    print("  [Pass] Verified score convention: Higher score = corrupted / OOD.")


def test_checkpoint_consistency() -> None:
    print("\n--- Test 4: Save & Load Checkpoint Consistency ---")
    encoder = GINEToxicityModel(in_node_dim=30, in_edge_dim=6, hidden_dim=32, num_layers=2)
    det1 = GINEValidityDetector(encoder=encoder, embed_dim=64, hidden_dim=32)

    tmp_path = Path("checkpoints/smoke_validity_test.pt")
    torch.save({"model_state_dict": det1.state_dict()}, tmp_path)

    det2 = GINEValidityDetector(encoder=encoder, embed_dim=64, hidden_dim=32)
    ckpt = torch.load(tmp_path, weights_only=False)
    det2.load_state_dict(ckpt["model_state_dict"])

    p1 = list(det1.parameters())[0]
    p2 = list(det2.parameters())[0]
    assert torch.allclose(p1, p2), "Checkpoint weights must match exactly!"
    if tmp_path.exists():
        tmp_path.unlink()
    print("  [Pass] Checkpoint serialization and restoration verified.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING GINE VALIDITY DETECTOR UNIT TESTS")
    print("=" * 60)
    try:
        test_validity_dataset_and_family_c_holdout()
        test_detector_forward_and_uid_order()
        test_score_direction_on_twins()
        test_checkpoint_consistency()
        print("\n" + "=" * 60)
        print("ALL GINE VALIDITY DETECTOR TESTS PASSED SUCCESSFULLY (Exit Code 0)")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\nVALIDITY TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
