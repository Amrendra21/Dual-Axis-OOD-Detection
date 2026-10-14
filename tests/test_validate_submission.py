"""
Automated Unit Tests for Submission Validator (validate_submission.py).

================================================================================
NOTICE ON TEST DATA:
All datasets in this file are STRICTLY SYNTHETIC FIXTURES generated solely for
unit testing prediction CSV validation logic.
They DO NOT represent biological assays, real molecules, or experimental findings.
================================================================================
"""

import os
import tempfile
import unittest
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

from evaluation.validate_submission import (
    validate_prediction_dataframe,
    validate_submission_file,
)


class TestValidateSubmission(unittest.TestCase):
    """Verify pre-flight validation rules for prediction CSV submissions."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_csv(self, filename: str, content: str) -> str:
        path = os.path.join(self.temp_dir.name, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_valid_submission(self):
        """Test that a well-formed prediction submission passes without errors."""
        # SYNTHETIC TEST FIXTURE ONLY
        valid_data = {
            "sample_id": ["mol_1", "mol_2", "mol_3", "mol_4"],
            "model_name": ["GCN_Energy", "GCN_Energy", "GCN_Energy", "GCN_Energy"],
            "benchmark": [
                "axis1_structural_validity", "axis1_structural_validity",
                "axis1_structural_validity", "axis1_structural_validity",
            ],
            "true_label": [0, 0, 1, 1],
            "ood_score": [0.12, 0.34, 0.88, 0.95],
            "seed": [42, 42, 42, 42],
        }
        df = pd.DataFrame(valid_data)
        is_valid, errors, warnings = validate_prediction_dataframe(df)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        # Should have orientation advisory warning
        self.assertTrue(any("SCORE ORIENTATION ADVISORY" in w for w in warnings))

    def test_missing_columns(self):
        """Test that missing required columns are flagged as errors."""
        # SYNTHETIC TEST FIXTURE - MISSING 'ood_score'
        bad_data = {
            "sample_id": ["mol_1", "mol_2"],
            "model_name": ["GCN", "GCN"],
            "benchmark": ["axis1", "axis1"],
            "true_label": [0, 1],
            "seed": [42, 42],
        }
        df = pd.DataFrame(bad_data)
        is_valid, errors, _ = validate_prediction_dataframe(df)

        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required column(s)" in e for e in errors))
        self.assertTrue(any("ood_score" in e for e in errors))

    def test_invalid_labels(self):
        """Test that non-binary labels (not 0 or 1) fail validation."""
        # SYNTHETIC TEST FIXTURE - LABELS CONTAIN 2 AND -1
        bad_data = {
            "sample_id": ["m1", "m2", "m3", "m4"],
            "model_name": ["M", "M", "M", "M"],
            "benchmark": ["b", "b", "b", "b"],
            "true_label": [0, 1, 2, -1],
            "ood_score": [0.1, 0.2, 0.3, 0.4],
            "seed": [42, 42, 42, 42],
        }
        df = pd.DataFrame(bad_data)
        is_valid, errors, _ = validate_prediction_dataframe(df)

        self.assertFalse(is_valid)
        self.assertTrue(any("true_label" in e and "invalid value" in e for e in errors))

    def test_nonnumeric_scores(self):
        """Test that unparseable non-numeric strings in ood_score fail validation."""
        # SYNTHETIC TEST FIXTURE - NONNUMERIC SCORE
        bad_data = {
            "sample_id": ["m1", "m2"],
            "model_name": ["M", "M"],
            "benchmark": ["b", "b"],
            "true_label": [0, 1],
            "ood_score": [0.1, "high_anomaly"],
            "seed": [42, 42],
        }
        df = pd.DataFrame(bad_data)
        is_valid, errors, _ = validate_prediction_dataframe(df)

        self.assertFalse(is_valid)
        self.assertTrue(any("non-numeric" in e for e in errors))

    def test_nan_and_infinity_scores(self):
        """Test that NaN and +/-Inf values in ood_score fail validation."""
        # SYNTHETIC TEST FIXTURE - NAN AND INF
        bad_data = {
            "sample_id": ["m1", "m2", "m3", "m4"],
            "model_name": ["M", "M", "M", "M"],
            "benchmark": ["b", "b", "b", "b"],
            "true_label": [0, 0, 1, 1],
            "ood_score": [0.1, np.nan, float("inf"), float("-inf")],
            "seed": [42, 42, 42, 42],
        }
        df = pd.DataFrame(bad_data)
        is_valid, errors, _ = validate_prediction_dataframe(df)

        self.assertFalse(is_valid)
        self.assertTrue(any("null/NaN" in e or "non-numeric" in e for e in errors))
        self.assertTrue(any("infinite" in e for e in errors))

    def test_empty_files(self):
        """Test that 0-byte files or files with 0 data rows fail validation."""
        # SYNTHETIC TEST FIXTURE - 0 BYTES
        empty_path = self._create_csv("zero_bytes.csv", "")
        self.assertFalse(validate_submission_file(empty_path, quiet=True))

        # SYNTHETIC TEST FIXTURE - HEADER ONLY
        header_path = self._create_csv(
            "header_only.csv",
            "sample_id,model_name,benchmark,true_label,ood_score,seed\n"
        )
        self.assertFalse(validate_submission_file(header_path, quiet=True))

    def test_missing_class_within_group(self):
        """Test that evaluation groups lacking either class 0 or class 1 fail."""
        # SYNTHETIC TEST FIXTURE - GROUP CONTAINS ONLY CLASS 0
        single_class_data = {
            "sample_id": ["m1", "m2", "m3"],
            "model_name": ["GCN", "GCN", "GCN"],
            "benchmark": ["axis1", "axis1", "axis1"],
            "true_label": [0, 0, 0],
            "ood_score": [0.1, 0.2, 0.3],
            "seed": [42, 42, 42],
        }
        df = pd.DataFrame(single_class_data)
        is_valid, errors, _ = validate_prediction_dataframe(df)

        self.assertFalse(is_valid)
        self.assertTrue(any("contains only class" in e for e in errors))

    def test_invalid_seeds(self):
        """Test that non-integer seeds fail validation."""
        # SYNTHETIC TEST FIXTURE - NON-INTEGER SEED
        bad_data = {
            "sample_id": ["m1", "m2"],
            "model_name": ["M", "M"],
            "benchmark": ["b", "b"],
            "true_label": [0, 1],
            "ood_score": [0.1, 0.8],
            "seed": ["seed_forty_two", 42.8],
        }
        df = pd.DataFrame(bad_data)
        is_valid, errors, _ = validate_prediction_dataframe(df)

        self.assertFalse(is_valid)
        self.assertTrue(any("invalid seed value" in e for e in errors))

    def test_extra_columns_allowed_with_warning(self):
        """Test that extra columns (like smiles) are permitted with a warning."""
        # SYNTHETIC TEST FIXTURE - EXTRA COLUMN 'smiles'
        data_with_extra = {
            "sample_id": ["m1", "m2"],
            "model_name": ["M", "M"],
            "benchmark": ["b", "b"],
            "true_label": [0, 1],
            "ood_score": [0.1, 0.8],
            "seed": [42, 42],
            "smiles": ["CC(=O)O", "c1ccccc1"],
        }
        df = pd.DataFrame(data_with_extra)
        is_valid, errors, warnings = validate_prediction_dataframe(df)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertTrue(any("Found extra column(s)" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
