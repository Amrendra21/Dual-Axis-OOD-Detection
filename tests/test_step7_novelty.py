"""
Unit and regression tests for Step 7: ECFP4 Tanimoto Novelty Estimator.

Tests:
1. Reference exclusion: validation and test molecules are strictly forbidden from fitting.
2. UID alignment and shape: score outputs match input UID length and order.
3. Score bounds and direction: novelty in [0.0, 1.0], exact match produces distance 0.0.
4. Determinism: identical SMILES produce bit-exact scores across independent calls.
"""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from step7_novelty.ecfp4_estimator import ECFP4TanimotoNoveltyEstimator


class TestECFP4NoveltyEstimator(unittest.TestCase):
    """Test suite for Step 7 ECFP4 Tanimoto Estimator."""

    @classmethod
    def setUpClass(cls):
        cls.master_path = Path("processed_v3/tables/master_table.csv")
        cls.master_df = pd.read_csv(cls.master_path).set_index("uid")
        cls.train_sub = cls.master_df[cls.master_df["set_name"] == "id_train"]

    def test_01_exclusion_of_non_id_train_splits(self):
        """Verify estimator rejects fitting on any split other than id_train."""
        est = ECFP4TanimotoNoveltyEstimator()
        forbidden_splits = [
            "id_val", "id_test", "novel_scaffold_val", "novel_scaffold_test",
            "novel_dataset_val", "novel_dataset_test", "corrupt_train", "corrupt_val", "corrupt_test"
        ]
        dummy_smiles = ["CC(=O)O"]
        dummy_uids = ["DUMMY-01"]

        for s in forbidden_splits:
            with self.assertRaises(ValueError, msg=f"Should raise ValueError on forbidden split '{s}'"):
                est.fit(dummy_smiles, dummy_uids, fit_split=s)

    def test_02_scoring_before_fitting_raises_error(self):
        """Verify scoring without fit raises RuntimeError."""
        est = ECFP4TanimotoNoveltyEstimator()
        with self.assertRaises(RuntimeError):
            est.score(["CC(=O)O"])

    def test_03_score_bounds_and_direction(self):
        """Verify score bounds in [0, 1] and identity produces distance 0.0."""
        est = ECFP4TanimotoNoveltyEstimator()
        sample_smiles = self.train_sub["smiles"].iloc[:20].tolist()
        sample_uids = self.train_sub.index[:20].tolist()
        est.fit(sample_smiles, sample_uids, fit_split="id_train")

        # Query the exact same molecules -> distance should be 0.0 (sim = 1.0)
        self_scores = est.score(sample_smiles[:5])
        self.assertEqual(len(self_scores), 5)
        for s in self_scores:
            self.assertAlmostEqual(s, 0.0, places=5)

        # Query different molecules
        val_sample = self.master_df[self.master_df["set_name"] == "id_val"]["smiles"].iloc[:10].tolist()
        val_scores = est.score(val_sample)
        self.assertEqual(len(val_scores), 10)
        self.assertTrue(np.all(val_scores >= 0.0) and np.all(val_scores <= 1.0))
        self.assertTrue(np.all(np.isfinite(val_scores)))

    def test_04_determinism(self):
        """Verify identical inputs yield bit-exact scores across runs."""
        sample_smiles = self.train_sub["smiles"].iloc[:50].tolist()
        sample_uids = self.train_sub.index[:50].tolist()

        est1 = ECFP4TanimotoNoveltyEstimator()
        est1.fit(sample_smiles, sample_uids, fit_split="id_train")

        est2 = ECFP4TanimotoNoveltyEstimator()
        est2.fit(sample_smiles, sample_uids, fit_split="id_train")

        val_sample = self.master_df[self.master_df["set_name"] == "id_val"]["smiles"].iloc[:20].tolist()
        s1 = est1.score(val_sample)
        s2 = est2.score(val_sample)
        np.testing.assert_array_equal(s1, s2)

    def test_05_uid_alignment(self):
        """Verify score_split_from_table preserves exact UID order."""
        est = ECFP4TanimotoNoveltyEstimator()
        sample_smiles = self.train_sub["smiles"].iloc[:30].tolist()
        sample_uids = self.train_sub.index[:30].tolist()
        est.fit(sample_smiles, sample_uids, fit_split="id_train")

        uids, scores = est.score_split_from_table("novel_scaffold_val")
        expected_uids = self.master_df[self.master_df["set_name"] == "novel_scaffold_val"].index.tolist()
        self.assertEqual(uids, expected_uids)
        self.assertEqual(len(scores), len(expected_uids))


if __name__ == "__main__":
    unittest.main()
