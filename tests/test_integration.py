"""
Integration test suite for Step 5: Dual-Axis Model Score Integration and Evaluation.

Tests:
1. Frozen component loading and dimension consistency (embed_dim = 256).
2. Multi-component scoring, UID preservation, and score finiteness.
3. Score direction verification (validity and novelty).
4. Standardized prediction CSV schema and content integrity.
5. Metric computation correctness (AUROC, AUPR, FPR95).
6. Test set authorization protection guard.
"""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data import get_dataloader
from src.evaluate import (
    compute_fpr95,
    compute_ood_metrics,
    load_frozen_components,
    score_split,
)
from src.run_final_eval import run_test_evaluation


class TestStep5Integration(unittest.TestCase):
    """Integration and regression test suite for Step 5."""

    @classmethod
    def setUpClass(cls):
        cls.toxicity_ckpt = "checkpoints/toxicity_gine/best_model.pt"
        cls.validity_ckpt = "checkpoints/validity_gine/best_validity_model.pt"
        cls.novelty_path = "checkpoints/novelty_mahal/estimator.npz"
        cls.tox_model, cls.val_detector, cls.nov_estimator = load_frozen_components(
            toxicity_ckpt_path=cls.toxicity_ckpt,
            validity_ckpt_path=cls.validity_ckpt,
            novelty_estimator_path=cls.novelty_path,
            device="cpu",
        )

    def test_01_component_loading_and_dimensions(self):
        """Verify all three frozen components load with identical 256d embedding spaces."""
        self.assertEqual(self.tox_model.embed_dim, 256)
        self.assertEqual(self.val_detector.embed_dim, 256)
        self.assertEqual(self.nov_estimator.embed_dim, 256)
        self.assertTrue(self.nov_estimator.is_fitted)

    def test_02_scoring_and_uid_alignment(self):
        """Verify score_split produces finite scores aligned 1-to-1 with UIDs."""
        loader = get_dataloader("id_val", batch_size=32, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            v_scores = self.val_detector.score_batch(batch).cpu().numpy()
            embs = self.tox_model.get_embeddings(batch).cpu().numpy()
            n_scores = self.nov_estimator.score(embs)

        self.assertEqual(len(batch.uid), len(v_scores))
        self.assertEqual(len(batch.uid), len(n_scores))
        self.assertTrue(np.all(np.isfinite(v_scores)))
        self.assertTrue(np.all(np.isfinite(n_scores)))
        self.assertTrue(np.all(v_scores >= 0.0) and np.all(v_scores <= 1.0))
        self.assertTrue(np.all(n_scores >= 0.0))

    def test_03_score_direction_verification(self):
        """Verify score directions: corrupted > valid, novel > ID train."""
        loader_valid = get_dataloader("id_val", batch_size=64, shuffle=False)
        loader_corrupt = get_dataloader("corrupt_val", batch_size=64, shuffle=False)

        batch_v = next(iter(loader_valid))
        batch_c = next(iter(loader_corrupt))

        with torch.no_grad():
            score_v = self.val_detector.score_batch(batch_v).mean().item()
            score_c = self.val_detector.score_batch(batch_c).mean().item()

        # Higher validity_score = more corrupted
        self.assertGreater(score_c, score_v)
        self.assertGreater(score_c, 0.7)
        self.assertLess(score_v, 0.3)

    def test_04_prediction_csv_schema_and_integrity(self):
        """Verify predictions CSV schema: uid, benchmark/split, true_label, validity_score, novelty_score."""
        expected_cols = ["uid", "benchmark/split", "true_label", "validity_score", "novelty_score"]

        for filename in ["val_predictions.csv", "test_predictions.csv"]:
            csv_path = Path("predictions") / filename
            self.assertTrue(csv_path.exists(), f"Missing file: {csv_path}")

            df = pd.read_csv(csv_path)
            self.assertEqual(list(df.columns), expected_cols)
            self.assertEqual(df.isna().sum().sum(), 0, f"Found NaNs in {filename}")
            self.assertTrue(set(df["true_label"].unique()).issubset({0, 1}))
            self.assertTrue(np.all(df["validity_score"] >= 0.0) and np.all(df["validity_score"] <= 1.0))
            self.assertTrue(np.all(df["novelty_score"] >= 0.0))

    def test_05_metric_computation(self):
        """Verify AUROC, AUPR, and FPR95 calculation behavior."""
        # Perfectly separable case
        y_true = np.array([0] * 100 + [1] * 100)
        y_score = np.array([0.1] * 100 + [0.9] * 100)
        metrics = compute_ood_metrics(y_true, y_score)

        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["aupr"], 1.0)
        self.assertAlmostEqual(metrics["fpr95"], 0.0)

        # Partially overlapping case
        y_score_overlap = np.array([0.1] * 90 + [0.8] * 10 + [0.2] * 5 + [0.9] * 95)
        fpr = compute_fpr95(y_true, y_score_overlap)
        self.assertLessEqual(fpr, 0.15)

    def test_06_final_eval_authorization_guard(self):
        """Verify run_test_evaluation blocks execution when authorize=False."""
        with self.assertRaises(SystemExit) as cm:
            run_test_evaluation(authorize=False)
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING STEP 5 INTEGRATION & EVALUATION TESTS")
    print("=" * 60)
    unittest.main()
