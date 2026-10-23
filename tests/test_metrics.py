"""
Automated Unit Tests for Evaluation Metrics (metrics.py).

================================================================================
NOTICE ON TEST DATA:
All datasets and tensors in this file are STRICTLY SYNTHETIC FIXTURES generated
solely for unit testing numerical software behavior.
They DO NOT represent real molecular graphs, biological assays, or experimental
findings from the Tox21 benchmark.
================================================================================
"""

import math
import unittest
import numpy as np

from evaluation.metrics import (
    compute_auroc,
    compute_aupr,
    compute_fpr95,
    compute_all_metrics,
    validate_inputs,
)


class TestMetrics(unittest.TestCase):
    """Unit tests verifying mathematical correctness and edge cases of metrics."""

    def test_perfect_separation(self):
        """
        Test that when OOD scores are strictly higher than all ID scores:
        - AUROC must equal 1.0
        - AUPR (OOD=positive) must equal 1.0
        - FPR95 must equal 0.0 (0% false alarms)
        """
        # SYNTHETIC TEST FIXTURE ONLY - NOT EXPERIMENTAL DATA
        SYNTHETIC_TEST_LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        SYNTHETIC_TEST_SCORES = np.array([0.1, 0.2, 0.25, 0.3, 0.8, 0.85, 0.9, 0.95])

        auroc = compute_auroc(SYNTHETIC_TEST_LABELS, SYNTHETIC_TEST_SCORES)
        aupr_out = compute_aupr(SYNTHETIC_TEST_LABELS, SYNTHETIC_TEST_SCORES, pos_label=1)
        fpr95 = compute_fpr95(SYNTHETIC_TEST_LABELS, SYNTHETIC_TEST_SCORES, target_tpr=0.95)

        self.assertAlmostEqual(auroc, 1.0, places=5)
        self.assertAlmostEqual(aupr_out, 1.0, places=5)
        self.assertAlmostEqual(fpr95, 0.0, places=5)

    def test_reversed_scores(self):
        """
        Test that when ID scores are strictly higher than all OOD scores (inverted score orientation):
        - AUROC must equal 0.0
        - FPR95 must equal 1.0 (100% false alarms to detect OOD samples)
        """
        # SYNTHETIC TEST FIXTURE ONLY - NOT EXPERIMENTAL DATA
        SYNTHETIC_TEST_LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Inverted scores: ID has higher scores than OOD
        SYNTHETIC_TEST_SCORES = np.array([0.8, 0.85, 0.9, 0.95, 0.1, 0.2, 0.25, 0.3])

        auroc = compute_auroc(SYNTHETIC_TEST_LABELS, SYNTHETIC_TEST_SCORES)
        fpr95 = compute_fpr95(SYNTHETIC_TEST_LABELS, SYNTHETIC_TEST_SCORES, target_tpr=0.95)

        self.assertAlmostEqual(auroc, 0.0, places=5)
        self.assertAlmostEqual(fpr95, 1.0, places=5)

    def test_random_and_tied_scores(self):
        """
        Test behavior when scores are identical across all samples:
        - AUROC must equal 0.5 (random chance)
        - Must not crash or divide by zero
        - FPR95 must be handled gracefully without exceptions
        """
        # SYNTHETIC TEST FIXTURE ONLY - ALL SCORES IDENTICAL (TIED)
        SYNTHETIC_TEST_LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        SYNTHETIC_TEST_SCORES = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        auroc = compute_auroc(SYNTHETIC_TEST_LABELS, SYNTHETIC_TEST_SCORES)
        fpr95 = compute_fpr95(SYNTHETIC_TEST_LABELS, SYNTHETIC_TEST_SCORES, target_tpr=0.95)
        all_metrics = compute_all_metrics(SYNTHETIC_TEST_LABELS, SYNTHETIC_TEST_SCORES)

        self.assertAlmostEqual(auroc, 0.5, places=5)
        self.assertIn("fpr95", all_metrics)
        self.assertTrue(0.0 <= fpr95 <= 1.0)

    def test_invalid_labels(self):
        """
        Test that invalid ground truth labels raise ValueError:
        - Non-binary labels (e.g. 2, -1)
        - Single class arrays (all 0s or all 1s)
        - String labels
        """
        # SYNTHETIC TEST FIXTURE ONLY - NON-BINARY LABELS
        labels_three_classes = [0, 1, 2, 0]
        scores_valid = [0.1, 0.5, 0.8, 0.2]
        with self.assertRaises(ValueError):
            validate_inputs(labels_three_classes, scores_valid)

        # SYNTHETIC TEST FIXTURE ONLY - NEGATIVE LABELS
        labels_negative = [-1, 0, -1, 0]
        with self.assertRaises(ValueError):
            validate_inputs(labels_negative, scores_valid)

        # SYNTHETIC TEST FIXTURE ONLY - SINGLE CLASS (ALL 0)
        labels_single_class_0 = [0, 0, 0, 0]
        with self.assertRaises(ValueError):
            validate_inputs(labels_single_class_0, scores_valid)

        # SYNTHETIC TEST FIXTURE ONLY - SINGLE CLASS (ALL 1)
        labels_single_class_1 = [1, 1, 1, 1]
        with self.assertRaises(ValueError):
            validate_inputs(labels_single_class_1, scores_valid)

        # SYNTHETIC TEST FIXTURE ONLY - STRING LABELS
        labels_strings = ["ID", "OOD", "ID", "OOD"]
        with self.assertRaises(ValueError):
            validate_inputs(labels_strings, scores_valid)

    def test_nan_values(self):
        """
        Test that NaN and Infinite values in either labels or scores raise ValueError.
        """
        # SYNTHETIC TEST FIXTURE ONLY - NAN IN LABELS
        labels_with_nan = [0, 1, float("nan"), 1]
        scores_valid = [0.1, 0.4, 0.7, 0.9]
        with self.assertRaises(ValueError):
            validate_inputs(labels_with_nan, scores_valid)

        # SYNTHETIC TEST FIXTURE ONLY - NAN IN SCORES
        labels_valid = [0, 0, 1, 1]
        scores_with_nan = [0.1, float("nan"), 0.7, 0.9]
        with self.assertRaises(ValueError):
            validate_inputs(labels_valid, scores_with_nan)

        # SYNTHETIC TEST FIXTURE ONLY - INF IN SCORES
        scores_with_inf = [0.1, float("inf"), 0.7, 0.9]
        with self.assertRaises(ValueError):
            validate_inputs(labels_valid, scores_with_inf)

    def test_mismatched_lengths(self):
        """
        Test that mismatched lengths between labels and scores raise ValueError.
        """
        # SYNTHETIC TEST FIXTURE ONLY - LENGTH MISMATCH
        labels_short = [0, 1, 0]
        scores_long = [0.1, 0.5, 0.8, 0.9, 0.95]
        with self.assertRaises(ValueError):
            validate_inputs(labels_short, scores_long)

    def test_empty_arrays(self):
        """
        Test that empty arrays raise ValueError.
        """
        # SYNTHETIC TEST FIXTURE ONLY - EMPTY ARRAYS
        with self.assertRaises(ValueError):
            validate_inputs([], [])

    def test_compute_all_metrics_structure(self):
        """
        Test that compute_all_metrics returns all documented keys with appropriate ranges.
        """
        # SYNTHETIC TEST FIXTURE ONLY
        SYNTHETIC_TEST_LABELS = np.array([0, 0, 0, 1, 1, 1])
        SYNTHETIC_TEST_SCORES = np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.9])

        metrics = compute_all_metrics(SYNTHETIC_TEST_LABELS, SYNTHETIC_TEST_SCORES)

        expected_keys = {"auroc", "aupr_out", "aupr_in", "fpr95", "fpr95_percent", "n_id", "n_ood", "n_total"}
        self.assertTrue(expected_keys.issubset(metrics.keys()))

        self.assertTrue(0.0 <= metrics["auroc"] <= 1.0)
        self.assertTrue(0.0 <= metrics["aupr_out"] <= 1.0)
        self.assertTrue(0.0 <= metrics["aupr_in"] <= 1.0)
        self.assertTrue(0.0 <= metrics["fpr95"] <= 1.0)
        self.assertTrue(0.0 <= metrics["fpr95_percent"] <= 100.0)
        self.assertEqual(metrics["n_id"], 3)
        self.assertEqual(metrics["n_ood"], 3)
        self.assertEqual(metrics["n_total"], 6)


if __name__ == "__main__":
    unittest.main()
