"""
Automated Unit Tests for Evaluation Runner (evaluate.py).

================================================================================
NOTICE ON TEST DATA:
All datasets in this file are STRICTLY SYNTHETIC FIXTURES generated solely for
unit testing CLI evaluation runner behavior and verifying non-fabrication policies.
================================================================================
"""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np

from evaluation.evaluate import evaluate_dataframe, run_evaluation


class TestEvaluateRunner(unittest.TestCase):
    """Verify evaluation runner mechanics and strict non-fabrication rules."""

    def test_evaluate_separate_benchmarks(self):
        """
        Verify that multiple benchmarks in the input file are evaluated separately,
        yielding distinct metric rows.
        """
        # SYNTHETIC TEST FIXTURE ONLY - MULTI-BENCHMARK DATASET
        synthetic_data = {
            "model_name": ["ModelA"] * 8,
            "benchmark": ["axis1_structural"] * 4 + ["axis2_semantic"] * 4,
            "label": [0, 0, 1, 1, 0, 0, 1, 1],
            "ood_score": [0.1, 0.2, 0.8, 0.9, 0.2, 0.3, 0.7, 0.85],
            "seed": [42] * 8,
        }
        df = pd.DataFrame(synthetic_data)
        results = evaluate_dataframe(df)

        self.assertFalse(results.empty)
        benchmarks_in_results = set(results["benchmark"].tolist())
        # Check both separate benchmarks were evaluated
        self.assertIn("axis1_structural", benchmarks_in_results)
        self.assertIn("axis2_semantic", benchmarks_in_results)

        # Both axes are present, so combined metric should be included
        self.assertIn("combined_dual_axis", benchmarks_in_results)

    def test_non_fabrication_when_axis_missing(self):
        """
        Verify that when only ONE axis is provided:
        - The runner evaluates the provided axis.
        - The runner NEVER fabricates, imputes, or guesses the missing axis or combined metric.
        """
        # SYNTHETIC TEST FIXTURE ONLY - SINGLE AXIS ONLY
        synthetic_data = {
            "model_name": ["ModelB"] * 4,
            "benchmark": ["axis1_structural"] * 4,
            "label": [0, 0, 1, 1],
            "ood_score": [0.1, 0.2, 0.8, 0.9],
            "seed": [42] * 4,
        }
        df = pd.DataFrame(synthetic_data)
        results = evaluate_dataframe(df)

        benchmarks_in_results = results["benchmark"].tolist()
        self.assertIn("axis1_structural", benchmarks_in_results)
        # MUST NOT fabricate axis2 or combined_dual_axis!
        self.assertNotIn("axis2_semantic", benchmarks_in_results)
        self.assertNotIn("combined_dual_axis", benchmarks_in_results)

    def test_rejection_of_nan_predictions(self):
        """
        Verify that NaNs in prediction labels or scores are rejected with ValueError
        to prevent downstream fabrication.
        """
        # SYNTHETIC TEST FIXTURE ONLY - CONTAINS NAN
        synthetic_data_nan = {
            "model_name": ["ModelC"] * 4,
            "benchmark": ["axis1_structural"] * 4,
            "label": [0, 1, 0, 1],
            "ood_score": [0.1, np.nan, 0.3, 0.9],
            "seed": [42] * 4,
        }
        df_nan = pd.DataFrame(synthetic_data_nan)
        with self.assertRaises(ValueError):
            evaluate_dataframe(df_nan)

    def test_member4_schema_automatic_score_selection(self):
        """
        Verify that Member 4 schema (uid, validity_score, novelty_score, benchmark)
        automatically selects validity_score for validity benchmarks and novelty_score
        for novelty benchmarks.
        """
        # SYNTHETIC TEST FIXTURE ONLY
        data = {
            "uid": [f"M4_{i}" for i in range(8)],
            "benchmark": ["validity/val/A"] * 4 + ["novelty/val/novel_scaffold"] * 4,
            "true_label": [0, 0, 1, 1, 0, 0, 1, 1],
            "validity_score": [0.1, 0.2, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5],
            "novelty_score": [0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.8, 0.9],
        }
        df = pd.DataFrame(data)
        results = evaluate_dataframe(df)

        self.assertFalse(results.empty)
        benchmarks = results["benchmark"].tolist()
        self.assertIn("validity/val/A", benchmarks)
        self.assertIn("novelty/val/novel_scaffold", benchmarks)
        self.assertIn("combined_dual_axis", benchmarks)

        # Validity benchmark should achieve AUROC 1.0 via validity_score
        val_row = results[results["benchmark"] == "validity/val/A"].iloc[0]
        self.assertGreater(val_row["auroc"], 0.9)
        self.assertEqual(val_row["score_col"], "validity_score")

        # Novelty benchmark should achieve AUROC 1.0 via novelty_score
        nov_row = results[results["benchmark"] == "novelty/val/novel_scaffold"].iloc[0]
        self.assertGreater(nov_row["auroc"], 0.9)
        self.assertEqual(nov_row["score_col"], "novelty_score")

    def test_member4_schema_benchmark_split_column(self):
        """
        Verify that Member 4 prediction files with literal 'benchmark/split' column
        are detected and correctly evaluated.
        """
        # SYNTHETIC TEST FIXTURE ONLY
        data = {
            "uid": [f"M4_{i}" for i in range(4)],
            "benchmark/split": ["validity_family_a"] * 4,
            "true_label": [0, 0, 1, 1],
            "validity_score": [0.15, 0.25, 0.85, 0.95],
        }
        df = pd.DataFrame(data)
        results = evaluate_dataframe(df)

        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]["benchmark"], "validity_family_a")
        self.assertEqual(results.iloc[0]["score_col"], "validity_score")
        self.assertGreater(results.iloc[0]["auroc"], 0.9)

    def test_member4_schema_separate_benchmark_and_split(self):
        """
        Verify that separate 'benchmark' and 'split' columns are combined into full tag.
        """
        # SYNTHETIC TEST FIXTURE ONLY
        data = {
            "uid": [f"M4_{i}" for i in range(4)],
            "benchmark": ["validity"] * 4,
            "split": ["val/B"] * 4,
            "true_label": [0, 0, 1, 1],
            "validity_score": [0.1, 0.2, 0.8, 0.9],
        }
        df = pd.DataFrame(data)
        results = evaluate_dataframe(df)

        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]["benchmark"], "validity/val/B")

    def test_rejection_of_ambiguous_score_selection(self):
        """
        When both validity_score and novelty_score are present and benchmark is ambiguous,
        evaluation must fail with ValueError rather than guessing.
        """
        # SYNTHETIC TEST FIXTURE ONLY - AMBIGUOUS BENCHMARK
        data = {
            "uid": ["m1", "m2", "m3", "m4"],
            "benchmark": ["unspecified_task"] * 4,
            "true_label": [0, 0, 1, 1],
            "validity_score": [0.1, 0.2, 0.8, 0.9],
            "novelty_score": [0.9, 0.8, 0.2, 0.1],
        }
        df = pd.DataFrame(data)
        with self.assertRaises(ValueError) as ctx:
            evaluate_dataframe(df)
        self.assertIn("Ambiguous score column selection", str(ctx.exception))

    def test_rejection_of_incompatible_score_column(self):
        """
        Verify that evaluating a validity benchmark with only novelty_score raises ValueError.
        """
        # SYNTHETIC TEST FIXTURE ONLY
        data = {
            "uid": ["m1", "m2", "m3", "m4"],
            "benchmark": ["validity/val/A"] * 4,
            "true_label": [0, 0, 1, 1],
            "novelty_score": [0.1, 0.2, 0.8, 0.9],
        }
        df = pd.DataFrame(data)
        with self.assertRaises(ValueError) as ctx:
            evaluate_dataframe(df)
        self.assertIn("Invalid score mapping", str(ctx.exception))

    def test_explicit_score_col_override(self):
        """
        Caller can explicitly specify score_col to override automatic resolution.
        """
        # SYNTHETIC TEST FIXTURE ONLY
        data = {
            "sample_id": ["m1", "m2", "m3", "m4"],
            "benchmark": ["validity/val/A"] * 4,
            "true_label": [0, 0, 1, 1],
            "custom_score": [0.1, 0.2, 0.8, 0.9],
        }
        df = pd.DataFrame(data)
        results = evaluate_dataframe(df, score_col="custom_score")
        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]["score_col"], "custom_score")
        self.assertGreater(results.iloc[0]["auroc"], 0.9)

    def test_rejection_of_nonexistent_explicit_score_col(self):
        """Specifying non-existent score_col raises ValueError."""
        # SYNTHETIC TEST FIXTURE ONLY
        data = {
            "sample_id": ["m1", "m2"],
            "benchmark": ["validity/val/A"] * 2,
            "true_label": [0, 1],
            "ood_score": [0.1, 0.9],
        }
        df = pd.DataFrame(data)
        with self.assertRaises(ValueError) as ctx:
            evaluate_dataframe(df, score_col="does_not_exist")
        self.assertIn("not found in DataFrame columns", str(ctx.exception))

    def test_ground_truth_label_alias_compatibility(self):
        """
        Verify that evaluate_dataframe supports 'ground_truth' as a label alias
        (e.g., from Member 4 prediction handoff) while maintaining backward compatibility.
        """
        # SYNTHETIC TEST FIXTURE ONLY - GROUND_TRUTH LABEL ALIAS
        data = {
            "uid": ["m1", "m2", "m3", "m4"],
            "benchmark": ["validity/val/A"] * 4,
            "ground_truth": [0, 0, 1, 1],
            "validity_score": [0.1, 0.2, 0.8, 0.9],
        }
        df = pd.DataFrame(data)
        results = evaluate_dataframe(df)
        self.assertFalse(results.empty)
        self.assertEqual(len(results), 1)
        self.assertGreater(results.iloc[0]["auroc"], 0.9)


if __name__ == "__main__":
    unittest.main()

