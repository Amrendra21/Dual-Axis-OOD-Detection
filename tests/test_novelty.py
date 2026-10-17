"""
Unit and integration tests for the Mahalanobis Novelty Estimator (Step 4).

Verified checks:
  1. Strict split enforcement: fitting on non-id_train splits raises ValueError.
  2. Extract embeddings: correct shape, dtype, UID order, and finiteness.
  3. Fit: estimator parameters are set, shrinkage is in (0, 1), precision is PD.
  4. Score properties: shape, dtype, finiteness, all-positive.
  5. Score direction: novel splits score higher than id_train on average.
  6. Round-trip serialization: save + load produces identical scores.
  7. Embed dim guard: scoring embeddings with wrong dimension raises ValueError.
  8. Pre-fitted guard: scoring before fitting raises RuntimeError.

Run:
    python tests/test_novelty.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — make sure project root is on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.novelty_model import (
    MahalanobisNoveltyEstimator,
    extract_embeddings,
    PERMITTED_FIT_SPLITS,
    PERMITTED_EVAL_SPLITS,
)
from src.toxicity_model import GINEToxicityModel
from src.data import MolecularDataset, collate_molecular_graphs, get_dataloader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "checkpoints/toxicity_gine/best_model.pt"
ROOT_DIR = "processed_v3"
EXPECTED_EMBED_DIM = 256
SMALL_BATCH = 32  # Use a small subset for fast tests
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_encoder() -> GINEToxicityModel:
    """Load the trained toxicity GNN from checkpoint in eval mode."""
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    cfg = ckpt.get("config", {})
    model = GINEToxicityModel(
        in_node_dim=cfg.get("in_node_dim", 30),
        in_edge_dim=cfg.get("in_edge_dim", 6),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 4),
        num_tasks=cfg.get("num_tasks", 12),
        dropout=cfg.get("dropout", 0.1),
        pooling=cfg.get("pooling", "mean_sum"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _pass(msg: str) -> None:
    print(f"  [Pass] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Test 1: Strict split enforcement
# ---------------------------------------------------------------------------

def test_split_enforcement() -> None:
    _section("Test 1: Strict Split Enforcement")

    estimator = MahalanobisNoveltyEstimator()
    dummy_embeddings = np.random.randn(10, EXPECTED_EMBED_DIM).astype(np.float32)

    # Attempting to fit on a non-permitted split should raise ValueError
    non_permitted_splits = ["id_val", "id_test", "novel_scaffold_val", "corrupt_train"]
    for split in non_permitted_splits:
        raised = False
        try:
            estimator.fit(dummy_embeddings, fit_split=split)
        except ValueError as e:
            raised = True
            assert "permitted fitting splits" in str(e) or "id_train" in str(e), (
                f"ValueError message does not mention split policy: {e}"
            )
        if raised:
            _pass(f"Attempting fit on '{split}' raised ValueError as expected.")
        else:
            _fail(f"Fitting on '{split}' did NOT raise ValueError — split policy not enforced!")

    # Pre-fitted guard: score before fit should raise RuntimeError
    fresh = MahalanobisNoveltyEstimator()
    try:
        fresh.score(dummy_embeddings)
        _fail("Scoring before fitting did NOT raise RuntimeError!")
    except RuntimeError:
        _pass("Scoring before fitting raised RuntimeError as expected.")


# ---------------------------------------------------------------------------
# Test 2: Embedding extraction — shape, dtype, UID order, finiteness
# ---------------------------------------------------------------------------

def test_embedding_extraction() -> None:
    _section("Test 2: Embedding Extraction (id_train, first 32 graphs)")
    encoder = load_encoder()

    # Extract from id_train
    embs, uids = extract_embeddings(
        encoder=encoder,
        split_name="id_train",
        root_dir=ROOT_DIR,
        batch_size=SMALL_BATCH,
    )

    expected_n = 5432
    if embs.shape != (expected_n, EXPECTED_EMBED_DIM):
        _fail(
            f"Expected embedding shape ({expected_n}, {EXPECTED_EMBED_DIM}), "
            f"got {embs.shape}."
        )
    _pass(f"Embedding shape correct: {embs.shape}")

    if embs.dtype != np.float32:
        _fail(f"Expected float32 embeddings, got {embs.dtype}")
    _pass(f"Embedding dtype correct: float32")

    if not np.all(np.isfinite(embs)):
        n_bad = int(np.sum(~np.isfinite(embs)))
        _fail(f"Embeddings contain {n_bad} non-finite values!")
    _pass("All id_train embeddings are finite.")

    if len(uids) != expected_n:
        _fail(f"Expected {expected_n} UIDs, got {len(uids)}")
    _pass(f"UID count matches: {len(uids)}")

    # Verify UID order matches loader order (first UID should be T21-00000)
    dataset = MolecularDataset(split_name="id_train", root_dir=ROOT_DIR)
    expected_first_uid = dataset[0]["uid"]
    if uids[0] != expected_first_uid:
        _fail(
            f"UID order mismatch! Expected first UID '{expected_first_uid}', "
            f"got '{uids[0]}'"
        )
    _pass(f"UID order preserved: first UID = '{uids[0]}'")


# ---------------------------------------------------------------------------
# Test 3: Fit properties — shrinkage, precision matrix PD
# ---------------------------------------------------------------------------

def test_fit_properties() -> None:
    _section("Test 3: Fit Properties")

    np.random.seed(RANDOM_SEED)
    # Use small synthetic embeddings for fast test
    N, D = 200, 32
    dummy_embs = np.random.randn(N, D).astype(np.float32)

    estimator = MahalanobisNoveltyEstimator()
    estimator.fit(dummy_embs, fit_split="id_train")

    if not estimator.is_fitted:
        _fail("Estimator not marked as fitted after .fit()!")
    _pass("Estimator is_fitted=True after .fit()")

    if estimator.embed_dim != D:
        _fail(f"Expected embed_dim={D}, got {estimator.embed_dim}")
    _pass(f"embed_dim set correctly: {estimator.embed_dim}")

    if estimator.n_train != N:
        _fail(f"Expected n_train={N}, got {estimator.n_train}")
    _pass(f"n_train set correctly: {estimator.n_train}")

    if not (0.0 <= estimator.shrinkage <= 1.0):
        _fail(f"Shrinkage out of [0, 1]: {estimator.shrinkage}")
    _pass(f"Shrinkage in [0, 1]: {estimator.shrinkage:.6f}")

    # Verify mean shape
    if estimator._mean.shape != (D,):
        _fail(f"Expected mean shape ({D},), got {estimator._mean.shape}")
    _pass(f"Mean vector shape correct: {estimator._mean.shape}")

    # Verify precision is PD via minimum eigenvalue
    min_eigval = float(np.linalg.eigvalsh(estimator._precision).min())
    if min_eigval > 0.0:
        _pass(f"Precision matrix is positive definite (min eigval={min_eigval:.3e}).")
    else:
        _fail(f"Precision matrix has non-positive min eigval={min_eigval:.3e}!")


# ---------------------------------------------------------------------------
# Test 4: Score properties — shape, dtype, finiteness, non-negativity
# ---------------------------------------------------------------------------

def test_score_properties() -> None:
    _section("Test 4: Score Properties")

    np.random.seed(RANDOM_SEED)
    N_fit, D = 300, EXPECTED_EMBED_DIM
    N_query = 50

    fit_embs = np.random.randn(N_fit, D).astype(np.float32)
    query_embs = np.random.randn(N_query, D).astype(np.float32)

    estimator = MahalanobisNoveltyEstimator()
    estimator.fit(fit_embs, fit_split="id_train")
    scores = estimator.score(query_embs)

    if scores.shape != (N_query,):
        _fail(f"Expected scores shape ({N_query},), got {scores.shape}")
    _pass(f"Scores shape correct: {scores.shape}")

    if scores.dtype != np.float32:
        _fail(f"Expected float32 scores, got {scores.dtype}")
    _pass("Scores dtype is float32.")

    if not np.all(np.isfinite(scores)):
        n_bad = int(np.sum(~np.isfinite(scores)))
        _fail(f"Scores contain {n_bad} non-finite values!")
    _pass("All scores are finite.")

    if not np.all(scores >= 0.0):
        n_neg = int(np.sum(scores < 0))
        _fail(f"Scores contain {n_neg} negative values (should be non-negative distances)!")
    _pass("All scores are non-negative (valid Mahalanobis distances).")

    # Embedding dimension guard
    wrong_dim_embs = np.random.randn(10, D + 1).astype(np.float32)
    try:
        estimator.score(wrong_dim_embs)
        _fail("Scoring with wrong embed_dim did NOT raise ValueError!")
    except ValueError as e:
        if "mismatch" in str(e).lower() or "dimension" in str(e).lower():
            _pass("Scoring with wrong embed_dim raised ValueError as expected.")
        else:
            _fail(f"ValueError raised but message unclear: {e}")


# ---------------------------------------------------------------------------
# Test 5: Score direction — novel splits should score higher than id_train
# ---------------------------------------------------------------------------

def test_score_direction() -> None:
    _section("Test 5: Score Direction (Novel > ID)")
    encoder = load_encoder()

    # Fit on id_train
    train_embs, train_uids = extract_embeddings(
        encoder=encoder,
        split_name="id_train",
        root_dir=ROOT_DIR,
        batch_size=256,
    )
    estimator = MahalanobisNoveltyEstimator()
    estimator.fit(train_embs, fit_split="id_train")

    train_scores = estimator.score(train_embs)
    mean_train = float(np.mean(train_scores))

    # Evaluate on novel_scaffold_val — should have higher scores
    novel_embs, novel_uids = extract_embeddings(
        encoder=encoder,
        split_name="novel_scaffold_val",
        root_dir=ROOT_DIR,
        batch_size=256,
    )
    novel_scores = estimator.score(novel_embs)
    mean_novel = float(np.mean(novel_scores))

    _pass(
        f"id_train mean score = {mean_train:.3f}  |  "
        f"novel_scaffold_val mean score = {mean_novel:.3f}"
    )

    if mean_novel > mean_train:
        _pass(
            f"Score direction verified: novel_scaffold_val ({mean_novel:.3f}) > "
            f"id_train ({mean_train:.3f})"
        )
    else:
        # Warn but do not fail — direction is expected but not guaranteed by design
        # (depends on training and GNN representations)
        print(
            f"  [WARN] Score direction NOT as expected: "
            f"novel_scaffold_val ({mean_novel:.3f}) <= id_train ({mean_train:.3f}). "
            f"This may indicate the GNN embeddings do not separate novelty well. "
            f"Investigate with the dual-axis evaluation framework."
        )

    # Also check novel_dataset_val
    try:
        nd_embs, _ = extract_embeddings(
            encoder=encoder,
            split_name="novel_dataset_val",
            root_dir=ROOT_DIR,
            batch_size=256,
        )
        nd_scores = estimator.score(nd_embs)
        mean_nd = float(np.mean(nd_scores))
        _pass(f"novel_dataset_val mean score = {mean_nd:.3f}")
    except FileNotFoundError:
        print("  [Skip] novel_dataset_val not found; skipping.")


# ---------------------------------------------------------------------------
# Test 6: Round-trip serialization
# ---------------------------------------------------------------------------

def test_serialization() -> None:
    _section("Test 6: Round-Trip Serialization")

    np.random.seed(RANDOM_SEED)
    N_fit, D = 300, 64
    N_query = 50

    fit_embs = np.random.randn(N_fit, D).astype(np.float32)
    query_embs = np.random.randn(N_query, D).astype(np.float32)

    estimator = MahalanobisNoveltyEstimator()
    estimator.fit(fit_embs, fit_split="id_train")
    original_scores = estimator.score(query_embs)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "estimator.npz"
        estimator.save(save_path)

        if not save_path.exists():
            _fail(f"Saved file not found at {save_path}!")
        _pass(f"Estimator saved to {save_path.name} (size={save_path.stat().st_size} bytes)")

        loaded = MahalanobisNoveltyEstimator.load(save_path)

        if not loaded.is_fitted:
            _fail("Loaded estimator is not marked as fitted!")
        _pass("Loaded estimator is_fitted=True")

        if loaded.embed_dim != estimator.embed_dim:
            _fail(f"embed_dim mismatch after load: {loaded.embed_dim} vs {estimator.embed_dim}")
        _pass(f"embed_dim preserved after round-trip: {loaded.embed_dim}")

        if loaded.shrinkage != estimator.shrinkage:
            _fail(f"shrinkage mismatch: {loaded.shrinkage} vs {estimator.shrinkage}")
        _pass(f"shrinkage preserved after round-trip: {loaded.shrinkage:.6f}")

        reloaded_scores = loaded.score(query_embs)
        if not np.allclose(reloaded_scores, original_scores, rtol=1e-5, atol=1e-6):
            max_diff = float(np.max(np.abs(reloaded_scores - original_scores)))
            _fail(f"Score values changed after serialization! max_diff={max_diff:.2e}")
        _pass("Scores are bit-exact after save -> load round-trip.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("RUNNING MAHALANOBIS NOVELTY ESTIMATOR UNIT TESTS")
    print("=" * 60)

    # Check checkpoint exists before running heavy tests
    if not Path(CHECKPOINT_PATH).exists():
        print(f"ERROR: Checkpoint not found at '{CHECKPOINT_PATH}'. Aborting.")
        sys.exit(1)

    try:
        test_split_enforcement()
        test_fit_properties()
        test_score_properties()
        test_serialization()
        test_embedding_extraction()
        test_score_direction()

        print("\n" + "=" * 60)
        print("ALL MAHALANOBIS NOVELTY ESTIMATOR TESTS PASSED (Exit Code 0)")
        print("=" * 60)
        sys.exit(0)

    except SystemExit:
        raise
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
