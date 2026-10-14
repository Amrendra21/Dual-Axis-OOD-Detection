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


if __name__ == "__main__":
    unittest.main()
