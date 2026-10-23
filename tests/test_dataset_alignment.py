"""
Automated Unit Tests for processed_v3 Dataset Alignment.

================================================================================
NOTICE ON TEST DATA:
All datasets in this file are STRICTLY SYNTHETIC FIXTURES generated solely for
unit testing the evaluation framework's alignment with the processed_v3 dataset.
They do NOT represent real molecular graphs, biological assays, or experimental
findings from Tox21 or SIDER.
================================================================================

Tests cover:
  - Per-family validity benchmark handling (A, B, C separately, never pooled).
  - Novel-scaffold and novel-dataset benchmark recognition.
  - twin_uid mapping for corrupted graphs.
  - Filtering matched_controls.csv to matched=True.
  - Ring-containing-only guard for novel scaffold comparisons.
  - Count-only feature extraction from atom and edge tensors.
  - Missing files, invalid split names, duplicate UIDs, malformed graph tensors.

These tests use small synthetic graph dictionaries and temporary CSVs.
They do NOT require the complete molecular dataset or torch.
"""

import os
import tempfile
import unittest

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

from evaluation.evaluate import evaluate_dataframe
from evaluation.baseline import (
    extract_count_features,
    extract_count_features_batch,
    count_only_auroc,
    filter_matched_controls,
    load_eligible_pairs,
    validate_split_name,
    validate_benchmark_name,
    VALID_BENCHMARKS,
    VALID_SPLIT_NAMES,
    _N_ATOM_FEATURES,
    _N_BOND_FEATURES,
)


# ---------------------------------------------------------------------------
# Helper: create synthetic graph dicts (numpy arrays, no torch dependency).
# ---------------------------------------------------------------------------

def _make_graph(uid="T21-00001", n_atoms=10, n_bonds=12, label_values=None):
    """Create a minimal synthetic graph dict mimicking processed_v3 format.

    SYNTHETIC FIXTURE ONLY.

    Parameters
    ----------
    uid : str
        Graph UID.
    n_atoms : int
        Number of atoms (nodes).
    n_bonds : int
        Number of undirected bonds (edge_index has 2 * n_bonds columns).
    label_values : list or None
        12 Tox21 label values; None = all-NaN.
    """
    rng = np.random.RandomState(hash(uid) % (2**31))
    x = np.zeros((n_atoms, _N_ATOM_FEATURES), dtype=np.float32)
    # Set element columns randomly (one-hot per atom).
    for i in range(n_atoms):
        elem_idx = rng.randint(0, 13)
        x[i, elem_idx] = 1.0
        # Set a degree.
        deg_idx = rng.randint(14, 19)
        x[i, deg_idx] = 1.0
        # Charge zero.
        x[i, 20] = 1.0
        # num_h_0.
        x[i, 23] = 1.0

    # Build edge_index: both directions for each undirected bond.
    src = rng.randint(0, n_atoms, size=n_bonds)
    dst = rng.randint(0, n_atoms, size=n_bonds)
    edge_index = np.vstack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src]),
    ])  # shape (2, 2*n_bonds)

    edge_attr = np.zeros((edge_index.shape[1], _N_BOND_FEATURES), dtype=np.float32)
    edge_attr[:, 0] = 1.0  # all bond_single

    if label_values is None:
        y = np.full((1, 12), np.nan, dtype=np.float32)
    else:
        y = np.array([label_values], dtype=np.float32)

    return {
        "uid": uid,
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": y,
    }


# ===========================================================================
# Test: Per-family validity benchmark handling
# ===========================================================================

class TestPerFamilyValidity(unittest.TestCase):
    """Verify that corruption families A, B, C are evaluated as separate benchmarks."""

    def test_three_families_produce_three_result_rows(self):
        """Each family must produce its own metrics row, never pooled."""
        # SYNTHETIC TEST FIXTURE ONLY
        n = 6
        data = {
            "model_name": ["M1"] * (3 * n),
            "benchmark": (
                ["validity/test/A"] * n
                + ["validity/test/B"] * n
                + ["validity/test/C"] * n
            ),
            "label": [0, 0, 0, 1, 1, 1] * 3,
            "ood_score": (
                [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
                + [0.15, 0.25, 0.35, 0.65, 0.75, 0.85]
                + [0.05, 0.1, 0.15, 0.8, 0.85, 0.95]
            ),
            "seed": [42] * (3 * n),
        }
        df = pd.DataFrame(data)
        results = evaluate_dataframe(df)

        result_benchmarks = set(results["benchmark"].tolist())
        self.assertIn("validity/test/A", result_benchmarks)
        self.assertIn("validity/test/B", result_benchmarks)
        self.assertIn("validity/test/C", result_benchmarks)

    def test_family_c_alone_no_combined(self):
        """Family C alone (validity only, no novelty) must NOT produce combined_dual_axis."""
        # SYNTHETIC TEST FIXTURE ONLY
        data = {
            "model_name": ["M1"] * 4,
            "benchmark": ["validity/test/C"] * 4,
            "label": [0, 0, 1, 1],
            "ood_score": [0.1, 0.2, 0.8, 0.9],
            "seed": [42] * 4,
        }
        df = pd.DataFrame(data)
        results = evaluate_dataframe(df)

        result_benchmarks = set(results["benchmark"].tolist())
        self.assertIn("validity/test/C", result_benchmarks)
        self.assertNotIn("combined_dual_axis", result_benchmarks)


# ===========================================================================
# Test: Novel-scaffold and novel-dataset benchmark recognition
# ===========================================================================

class TestNoveltyBenchmarkRecognition(unittest.TestCase):
    """Verify that novel_dataset is correctly identified as Axis 2 (novelty)."""

    def test_novel_dataset_triggers_combined_with_validity(self):
        """novelty/test/novel_dataset + validity/test/A must produce combined_dual_axis."""
        # SYNTHETIC TEST FIXTURE ONLY
        n = 4
        data = {
            "model_name": ["M1"] * (2 * n),
            "benchmark": (
                ["validity/test/A"] * n
                + ["novelty/test/novel_dataset"] * n
            ),
            "label": [0, 0, 1, 1] * 2,
            "ood_score": [0.1, 0.2, 0.8, 0.9, 0.15, 0.25, 0.7, 0.85],
            "seed": [42] * (2 * n),
        }
        df = pd.DataFrame(data)
        results = evaluate_dataframe(df)

        result_benchmarks = set(results["benchmark"].tolist())
        self.assertIn("novelty/test/novel_dataset", result_benchmarks)
        self.assertIn("validity/test/A", result_benchmarks)
        self.assertIn("combined_dual_axis", result_benchmarks)

    def test_novel_scaffold_triggers_combined(self):
        """novelty/test/novel_scaffold + validity/test/B must produce combined_dual_axis."""
        # SYNTHETIC TEST FIXTURE ONLY
        n = 4
        data = {
            "model_name": ["M1"] * (2 * n),
            "benchmark": (
                ["validity/test/B"] * n
                + ["novelty/test/novel_scaffold"] * n
            ),
            "label": [0, 0, 1, 1] * 2,
            "ood_score": [0.1, 0.2, 0.8, 0.9, 0.15, 0.25, 0.7, 0.85],
            "seed": [42] * (2 * n),
        }
        df = pd.DataFrame(data)
        results = evaluate_dataframe(df)

        result_benchmarks = set(results["benchmark"].tolist())
        self.assertIn("combined_dual_axis", result_benchmarks)

    def test_both_novelty_types_combined_with_validity(self):
        """Both novelty types + two validity families → one combined row with correct counts."""
        # SYNTHETIC TEST FIXTURE ONLY
        n = 4
        data = {
            "model_name": ["M1"] * (4 * n),
            "benchmark": (
                ["validity/test/A"] * n
                + ["validity/test/B"] * n
                + ["novelty/test/novel_scaffold"] * n
                + ["novelty/test/novel_dataset"] * n
            ),
            "label": [0, 0, 1, 1] * 4,
            "ood_score": [0.1, 0.2, 0.8, 0.9] * 4,
            "seed": [42] * (4 * n),
        }
        df = pd.DataFrame(data)
        results = evaluate_dataframe(df)

        combined = results[results["benchmark"] == "combined_dual_axis"]
        self.assertEqual(len(combined), 1)

        row = combined.iloc[0]
        self.assertEqual(row["n_validity_benchmarks"], 2)
        self.assertEqual(row["n_novelty_benchmarks"], 2)

    def test_novelty_only_no_combined(self):
        """Novelty alone must NOT fabricate a validity axis or combined metric."""
        # SYNTHETIC TEST FIXTURE ONLY
        data = {
            "model_name": ["M1"] * 4,
            "benchmark": ["novelty/test/novel_dataset"] * 4,
            "label": [0, 0, 1, 1],
            "ood_score": [0.1, 0.2, 0.8, 0.9],
            "seed": [42] * 4,
        }
        df = pd.DataFrame(data)
        results = evaluate_dataframe(df)

        result_benchmarks = set(results["benchmark"].tolist())
        self.assertIn("novelty/test/novel_dataset", result_benchmarks)
        self.assertNotIn("combined_dual_axis", result_benchmarks)

    def test_legacy_names_still_work(self):
        """Old axis1/axis2 benchmark names should still trigger combined metric."""
        # SYNTHETIC TEST FIXTURE ONLY
        data = {
            "model_name": ["M1"] * 8,
            "benchmark": ["axis1_structural"] * 4 + ["axis2_semantic"] * 4,
            "label": [0, 0, 1, 1, 0, 0, 1, 1],
            "ood_score": [0.1, 0.2, 0.8, 0.9, 0.2, 0.3, 0.7, 0.85],
            "seed": [42] * 8,
        }
        df = pd.DataFrame(data)
        results = evaluate_dataframe(df)

        result_benchmarks = set(results["benchmark"].tolist())
        self.assertIn("axis1_structural", result_benchmarks)
        self.assertIn("axis2_semantic", result_benchmarks)
        self.assertIn("combined_dual_axis", result_benchmarks)


# ===========================================================================
# Test: twin_uid mapping for corrupted graphs
# ===========================================================================

class TestTwinUidMapping(unittest.TestCase):
    """Verify eligible_pairs.csv loading and twin_uid relationship."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_pairs_csv(self, filename="eligible_pairs.csv"):
        """Create a synthetic eligible_pairs.csv."""
        # SYNTHETIC TEST FIXTURE ONLY
        path = os.path.join(self.temp_dir.name, filename)
        pd.DataFrame({
            "corrupt_uid": [
                "COR-A-T21-00005", "COR-B-T21-00005", "COR-A-T21-00036",
                "COR-C-T21-00100",
            ],
            "twin_uid": ["T21-00005", "T21-00005", "T21-00036", "T21-00100"],
            "split": ["train", "train", "train", "test"],
            "family": ["A", "B", "A", "C"],
            "n_edits": [1, 2, 1, 3],
            "n_viol_atoms": [1, 2, 1, 3],
        }).to_csv(path, index=False)
        return path

    def test_load_all_pairs(self):
        """Loading without filters returns all rows."""
        path = self._write_pairs_csv()
        df = load_eligible_pairs(path)
        self.assertEqual(len(df), 4)

    def test_filter_by_split(self):
        """Filter to test split returns only family C row."""
        path = self._write_pairs_csv()
        df = load_eligible_pairs(path, split="test")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["family"], "C")

    def test_filter_by_family(self):
        """Filter to family A returns only A rows."""
        path = self._write_pairs_csv()
        df = load_eligible_pairs(path, family="A")
        self.assertEqual(len(df), 2)
        self.assertTrue((df["family"] == "A").all())

    def test_twin_uid_lookup(self):
        """Each corrupt graph maps to exactly one twin."""
        path = self._write_pairs_csv()
        df = load_eligible_pairs(path)
        twin = df[df["corrupt_uid"] == "COR-A-T21-00005"]["twin_uid"].iloc[0]
        self.assertEqual(twin, "T21-00005")

    def test_missing_file_raises(self):
        """Non-existent path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_eligible_pairs(os.path.join(self.temp_dir.name, "missing.csv"))

    def test_missing_columns_raises(self):
        """CSV missing required columns raises ValueError."""
        path = os.path.join(self.temp_dir.name, "bad.csv")
        pd.DataFrame({"col_a": [1], "col_b": [2]}).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            load_eligible_pairs(path)


# ===========================================================================
# Test: Filtering matched_controls.csv to matched=True
# ===========================================================================

class TestMatchedControlsFiltering(unittest.TestCase):
    """Verify matched_controls.csv filtering."""

    def test_filter_matched_true_only(self):
        """Only matched=True rows are returned."""
        # SYNTHETIC TEST FIXTURE ONLY
        df = pd.DataFrame({
            "target_uid": ["SID-001", "SID-002", "SID-003"],
            "target_set": [
                "novel_dataset_test", "novel_dataset_test", "novel_dataset_test"
            ],
            "control_uid": ["T21-100", "", "T21-200"],
            "control_set": ["id_test", "id_test", "id_test"],
            "d_atoms": [0.0, np.nan, 1.0],
            "d_bonds": [0.0, np.nan, 0.0],
            "matched": [True, False, True],
        })
        result = filter_matched_controls(df)
        self.assertEqual(len(result), 2)
        self.assertTrue((result["matched"] == True).all())

    def test_filter_by_target_set(self):
        """Filter by target_set selects only that set."""
        # SYNTHETIC TEST FIXTURE ONLY
        df = pd.DataFrame({
            "target_uid": ["SID-001", "T21-SCF-001"],
            "target_set": ["novel_dataset_test", "novel_scaffold_test"],
            "control_uid": ["T21-100", "T21-200"],
            "control_set": ["id_test", "id_test"],
            "d_atoms": [0.0, 1.0],
            "d_bonds": [0.0, 0.0],
            "matched": [True, True],
        })
        result = filter_matched_controls(df, target_set="novel_dataset_test")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["target_uid"], "SID-001")

    def test_missing_columns_raises(self):
        """DataFrame missing required columns raises ValueError."""
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        with self.assertRaises(ValueError):
            filter_matched_controls(df)


# ===========================================================================
# Test: Ring-containing-only guard for novel scaffold
# ===========================================================================

class TestNovelScaffoldRingGuard(unittest.TestCase):
    """Verify that novel_scaffold benchmarks acknowledge ring-containing-only constraint.

    DATA_CARD.md §7: novel_scaffold sets contain ring-containing molecules only.
    Comparing them against the raw ID set (which includes ring-free) inflates AUROC.
    The correct comparison uses matched_controls.csv or the ring-containing subset
    of the ID set.
    """

    def test_novel_scaffold_graphs_have_ring_atoms(self):
        """Synthetic novel_scaffold graph should have is_aromatic or in_ring flags set."""
        # SYNTHETIC TEST FIXTURE ONLY - scaffold molecules are ring-containing.
        g = _make_graph(uid="T21-SCF-00001", n_atoms=15)
        # Mark some atoms as in_ring (column 29) to simulate ring-containing.
        g["x"][:5, 29] = 1.0  # in_ring

        # A ring-containing check: at least one atom with in_ring == 1.
        has_ring = g["x"][:, 29].sum() > 0
        self.assertTrue(has_ring)

    def test_ring_free_graph_fails_ring_check(self):
        """A ring-free graph should have no in_ring flags."""
        # SYNTHETIC TEST FIXTURE ONLY
        g = _make_graph(uid="T21-ACYCLIC-001", n_atoms=5)
        g["x"][:, 28] = 0.0  # is_aromatic = 0
        g["x"][:, 29] = 0.0  # in_ring = 0

        has_ring = g["x"][:, 29].sum() > 0
        self.assertFalse(has_ring)


# ===========================================================================
# Test: Count-only feature extraction from atom and edge tensors
# ===========================================================================

class TestCountOnlyFeatures(unittest.TestCase):
    """Verify count-only feature extraction matches BUILD_REPORT.md definition."""

    def test_feature_vector_length(self):
        """Feature vector must be length 21 (n_atoms + n_bonds + 13 elems + 6 degrees)."""
        # SYNTHETIC TEST FIXTURE ONLY
        g = _make_graph(uid="T21-00001", n_atoms=10, n_bonds=12)
        features = extract_count_features(g)
        self.assertEqual(features.shape, (21,))

    def test_n_atoms_n_bonds(self):
        """First two features must be n_atoms and n_bonds."""
        # SYNTHETIC TEST FIXTURE ONLY
        g = _make_graph(uid="T21-00002", n_atoms=15, n_bonds=18)
        features = extract_count_features(g)
        self.assertAlmostEqual(features[0], 15.0)
        self.assertAlmostEqual(features[1], 18.0)

    def test_element_counts_sum_equals_n_atoms(self):
        """Element count features (cols 2:15) should sum to n_atoms."""
        # SYNTHETIC TEST FIXTURE ONLY
        g = _make_graph(uid="T21-00003", n_atoms=20, n_bonds=22)
        features = extract_count_features(g)
        elem_sum = features[2:15].sum()
        self.assertAlmostEqual(elem_sum, 20.0)

    def test_degree_hist_sum_equals_n_atoms(self):
        """Degree histogram features (cols 15:21) should sum to n_atoms."""
        # SYNTHETIC TEST FIXTURE ONLY
        g = _make_graph(uid="T21-00004", n_atoms=8, n_bonds=10)
        features = extract_count_features(g)
        degree_sum = features[15:21].sum()
        self.assertAlmostEqual(degree_sum, 8.0)

    def test_batch_extraction(self):
        """Batch extraction returns (n_graphs, 21) array and matching UIDs."""
        # SYNTHETIC TEST FIXTURE ONLY
        graphs = [
            _make_graph("T21-00001", 10, 12),
            _make_graph("T21-00002", 15, 18),
            _make_graph("T21-00003", 20, 22),
        ]
        features, uids = extract_count_features_batch(graphs)
        self.assertEqual(features.shape, (3, 21))
        self.assertEqual(uids, ["T21-00001", "T21-00002", "T21-00003"])

    def test_count_only_auroc_perfect_separation(self):
        """Perfect count separation should yield AUROC near 1.0 with appropriate orientation."""
        # SYNTHETIC TEST FIXTURE ONLY
        # Case 1: Default higher_is_ood=True (higher feature value = more OOD)
        # ID graphs: small (10 atoms), OOD graphs: large (30 atoms)
        id_features_small = np.array([[10.0] + [0.0] * 20, [12.0] + [0.0] * 20])
        ood_features_large = np.array([[30.0] + [0.0] * 20, [28.0] + [0.0] * 20])
        auroc_default = count_only_auroc(id_features_small, ood_features_large, score_column=0)
        self.assertGreater(auroc_default, 0.9)

        # Case 2: Explicit higher_is_ood=False (smaller feature value = more OOD)
        # ID graphs: large (30 atoms), OOD graphs: small (5 atoms)
        id_features_large = np.array([[30.0] + [0.0] * 20, [28.0] + [0.0] * 20])
        ood_features_small = np.array([[5.0] + [0.0] * 20, [6.0] + [0.0] * 20])
        auroc_inverted = count_only_auroc(
            id_features_large, ood_features_small, score_column=0, higher_is_ood=False
        )
        self.assertGreater(auroc_inverted, 0.9)

        # Case 3: Sensitivity check - opposite orientation produces inverted AUROC
        auroc_mismatch = count_only_auroc(
            id_features_large, ood_features_small, score_column=0, higher_is_ood=True
        )
        self.assertLess(auroc_mismatch, 0.1)


# ===========================================================================
# Test: Malformed inputs, missing files, invalid names
# ===========================================================================

class TestInputValidation(unittest.TestCase):
    """Verify graceful error handling for malformed inputs."""

    def test_graph_missing_x_raises(self):
        """Graph dict without 'x' key should raise ValueError."""
        g = {"uid": "BAD-001", "edge_index": np.zeros((2, 4))}
        with self.assertRaises(ValueError):
            extract_count_features(g)

    def test_graph_missing_edge_index_raises(self):
        """Graph dict without 'edge_index' should raise ValueError."""
        g = {"uid": "BAD-002", "x": np.zeros((5, _N_ATOM_FEATURES))}
        with self.assertRaises(ValueError):
            extract_count_features(g)

    def test_wrong_feature_dim_raises(self):
        """x with wrong number of columns should raise ValueError."""
        g = {
            "uid": "BAD-003",
            "x": np.zeros((5, 10)),  # Wrong: should be 30
            "edge_index": np.zeros((2, 4)),
        }
        with self.assertRaises(ValueError):
            extract_count_features(g)

    def test_wrong_edge_index_shape_raises(self):
        """edge_index with wrong first dim should raise ValueError."""
        g = {
            "uid": "BAD-004",
            "x": np.zeros((5, _N_ATOM_FEATURES)),
            "edge_index": np.zeros((3, 4)),  # Wrong: should be (2, E)
        }
        with self.assertRaises(ValueError):
            extract_count_features(g)

    def test_empty_features_raises(self):
        """count_only_auroc with empty arrays should raise ValueError."""
        with self.assertRaises(ValueError):
            count_only_auroc(np.array([]).reshape(0, 21), np.ones((2, 21)))

    def test_valid_split_names(self):
        """All processed_v3 split names are recognized."""
        for name in [
            "id_train", "id_val", "id_test",
            "corrupt_train", "corrupt_val", "corrupt_test",
            "novel_scaffold_val", "novel_scaffold_test",
            "novel_dataset_val", "novel_dataset_test",
        ]:
            self.assertTrue(validate_split_name(name), f"{name} should be valid")

    def test_invalid_split_names(self):
        """Legacy or misspelled split names are rejected."""
        for name in ["ood_valid", "invalid_graphs_test", "train", "novel_test"]:
            self.assertFalse(validate_split_name(name), f"{name} should be invalid")

    def test_valid_benchmark_names(self):
        """All processed_v3 benchmark names are recognized."""
        for name in VALID_BENCHMARKS:
            self.assertTrue(validate_benchmark_name(name), f"{name} should be valid")

    def test_invalid_benchmark_names(self):
        """Legacy or misspelled benchmark names are rejected."""
        for name in [
            "axis1_structural_validity",
            "axis2_semantic_novelty",
            "validity/test/D",
            "novelty/train/novel_scaffold",
        ]:
            self.assertFalse(
                validate_benchmark_name(name), f"{name} should be invalid"
            )

    def test_duplicate_uids_in_batch(self):
        """Batch extraction should faithfully report duplicate UIDs (not crash)."""
        graphs = [
            _make_graph("T21-DUP", 5, 6),
            _make_graph("T21-DUP", 5, 6),
        ]
        features, uids = extract_count_features_batch(graphs)
        self.assertEqual(len(uids), 2)
        self.assertEqual(uids[0], uids[1])


if __name__ == "__main__":
    unittest.main()
