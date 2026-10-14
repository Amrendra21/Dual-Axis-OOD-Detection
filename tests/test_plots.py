"""
Automated Unit Tests for Plotting Module (plots.py).

================================================================================
NOTICE ON TEST DATA:
All datasets in this file are STRICTLY SYNTHETIC FIXTURES generated solely for
unit testing matplotlib plotting routines and verifying file generation.
================================================================================
"""

import os
import tempfile
import unittest
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

from evaluation.plots import (
    plot_roc_curve,
    plot_multi_roc_curves,
    plot_metric_comparison,
)


class TestPlots(unittest.TestCase):
    """Verify that plotting utilities generate valid figure files without fabrication."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_plot_roc_curve(self):
        """Test single ROC curve generation."""
        # SYNTHETIC TEST FIXTURE ONLY
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        out_file = os.path.join(self.temp_dir.name, "test_roc.png")

        fig = plot_roc_curve(y_true, y_score, output_path=out_file)
        self.assertTrue(os.path.exists(out_file))
        self.assertGreater(os.path.getsize(out_file), 1000)

    def test_plot_multi_roc_curves(self):
        """Test multi-ROC curve generation."""
        # SYNTHETIC TEST FIXTURE ONLY
        curves_dict = {
            "Axis 1 (Structural)": (np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])),
            "Axis 2 (Semantic)": (np.array([0, 0, 1, 1]), np.array([0.2, 0.3, 0.7, 0.85])),
        }
        out_file = os.path.join(self.temp_dir.name, "test_multi_roc.png")

        fig = plot_multi_roc_curves(curves_dict, output_path=out_file)
        self.assertTrue(os.path.exists(out_file))
        self.assertGreater(os.path.getsize(out_file), 1000)

    def test_plot_metric_comparison(self):
        """Test metric comparison bar chart generation from summary DataFrame."""
        # SYNTHETIC TEST FIXTURE ONLY
        summary_df = pd.DataFrame({
            "benchmark": ["axis1_structural", "axis2_semantic", "combined_dual_axis"],
            "auroc": [0.92, 0.85, 0.885],
            "aupr_out": [0.89, 0.81, 0.85],
            "fpr95": [0.12, 0.25, 0.185],
        })
        out_file = os.path.join(self.temp_dir.name, "test_comparison.png")

        fig = plot_metric_comparison(summary_df, output_path=out_file)
        self.assertTrue(os.path.exists(out_file))
        self.assertGreater(os.path.getsize(out_file), 1000)


if __name__ == "__main__":
    unittest.main()
