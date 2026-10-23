"""
Count-Only Baseline for Dual-Axis OOD Detection.

Implements the shortcut audit's count-only feature extraction and evaluation,
as defined in processed_v3/BUILD_REPORT.md and audits/audit_results.json.

Count-only features use ONLY graph-level scalar summaries — no bond types,
no connectivity, no learned representations.  Any model that cannot beat
the count-only AUROC on a given benchmark is relying on graph-size or
element-composition cues rather than structural understanding.

Feature definition (verified against processed_v3/feature_schema.json):
  - n_atoms:        number of nodes = x.shape[0]
  - n_bonds:        number of undirected edges = edge_index.shape[1] // 2
  - element_counts: x[:, 0:13].sum(0)  →  13 values (elem_C .. elem_other)
  - degree_hist:    x[:, 13:19].sum(0)  →  6 values (degree_0 .. degree_5plus)
  Total: 21 scalar features per graph.

Reference AUROC values from BUILD_REPORT.md (seed 42, processed_v3):
  These are audit outputs from Member 2's build pipeline.  They are documented
  here as reference baselines, NOT as independently reproduced results.

  Validity (count-only AUROC, threshold=0.65):
    train/A = 0.518   val/A = 0.493   test/A = 0.489   (no shortcut)
    train/B = 0.721   val/B = 0.652   test/B = 0.652   (shortcut / borderline)
                                      test/C = 0.798   (structural shortcut)

  Novelty (count-only AUROC):
    val/novel_scaffold  = 0.752   test/novel_scaffold  = 0.755
    val/novel_dataset   = 0.728   test/novel_dataset   = 0.693

  Novelty size-matched (using matched_controls.csv):
    val/novel_scaffold  = 0.689   test/novel_scaffold  = 0.646
    val/novel_dataset   = 0.664   test/novel_dataset   = 0.629

Strict Non-Fabrication Policy:
  - This module never hardcodes the above values as model outputs.
  - All AUROC values returned by functions here are computed from actual
    feature vectors, not copied from the audit.
  - Reference values may be loaded from audits/audit_results.json for
    comparison, clearly labelled as "BUILD_REPORT reference".
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

# Feature schema column ranges (from processed_v3/feature_schema.json).
# atom_features[0:13]  = element one-hots (elem_C .. elem_other)
# atom_features[13:19] = degree one-hots  (degree_0 .. degree_5plus)
# atom_features[19:23] = charge one-hots  (charge_m1 .. charge_other)
# atom_features[23:28] = num_h one-hots   (num_h_0 .. num_h_4plus)
# atom_features[28]    = is_aromatic
# atom_features[29]    = in_ring
_ELEM_SLICE = slice(0, 13)   # 13 columns
_DEGREE_SLICE = slice(13, 19)  # 6 columns
_N_ATOM_FEATURES = 30
_N_BOND_FEATURES = 6

# Reference AUROC values from BUILD_REPORT.md shortcut audit (seed 42).
# These are NOT independently computed — they are Member 2's audit outputs.
AUDIT_REFERENCE_AUROCS = {
    "validity": {
        "train/A": 0.5185, "train/B": 0.7206, "train/ALL": 0.6248,
        "val/A": 0.4927,   "val/B": 0.6523,   "val/ALL": 0.5883,
        "test/A": 0.4891,  "test/B": 0.6518,  "test/C": 0.7978,
        "test/ALL": 0.6263,
    },
    "novelty": {
        "val/novel_scaffold": 0.7518,  "val/novel_dataset": 0.7280,
        "test/novel_scaffold": 0.7545, "test/novel_dataset": 0.6931,
    },
    "novelty_size_matched": {
        "val/novel_scaffold": 0.6885,  "val/novel_dataset": 0.6637,
        "test/novel_scaffold": 0.6459, "test/novel_dataset": 0.6292,
    },
    "shortcut_threshold": 0.65,
}


def extract_count_features(graph: Dict[str, Any]) -> np.ndarray:
    """Extract count-only features from a single processed_v3 graph dict.

    Parameters
    ----------
    graph : dict
        A graph dictionary as loaded from ``torch.load(path, weights_only=True)``.
        Must contain keys ``"x"`` (atom features, shape [N, 30]) and
        ``"edge_index"`` (shape [2, E]).

    Returns
    -------
    np.ndarray
        1-D array of 21 scalar features:
        [n_atoms, n_bonds, elem_C, ..., elem_other(13), degree_0, ..., degree_5plus(6)]

    Raises
    ------
    ValueError
        If the graph dict is missing required keys or has unexpected shapes.
    """
    if "x" not in graph:
        raise ValueError("Graph dict missing required key 'x' (atom features).")
    if "edge_index" not in graph:
        raise ValueError("Graph dict missing required key 'edge_index'.")

    # Support both torch tensors and numpy arrays.
    x = graph["x"]
    edge_index = graph["edge_index"]

    # Convert to numpy if necessary (torch tensors).
    if hasattr(x, "numpy"):
        x = x.numpy()
    if hasattr(edge_index, "numpy"):
        edge_index = edge_index.numpy()

    x = np.asarray(x, dtype=np.float64)
    edge_index = np.asarray(edge_index)

    if x.ndim != 2 or x.shape[1] != _N_ATOM_FEATURES:
        raise ValueError(
            f"Expected x.shape = [N, {_N_ATOM_FEATURES}], got {x.shape}."
        )
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"Expected edge_index.shape = [2, E], got {edge_index.shape}."
        )

    n_atoms = float(x.shape[0])
    # edge_index stores both directions; undirected bond count = E / 2.
    n_bonds = float(edge_index.shape[1]) / 2.0

    element_counts = x[:, _ELEM_SLICE].sum(axis=0)  # shape (13,)
    degree_hist = x[:, _DEGREE_SLICE].sum(axis=0)    # shape (6,)

    features = np.concatenate([[n_atoms, n_bonds], element_counts, degree_hist])
    return features  # shape (21,)


def extract_count_features_batch(
    graphs: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str]]:
    """Extract count-only features from a list of graph dicts.

    Parameters
    ----------
    graphs : list of dict
        Graph dicts as returned by ``torch.load(...)``.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        - features: 2-D array of shape ``(n_graphs, 21)``
        - uids: list of uid strings (empty string if uid is missing)
    """
    features_list = []
    uids = []
    for g in graphs:
        features_list.append(extract_count_features(g))
        uids.append(str(g.get("uid", "")))
    return np.vstack(features_list), uids


def count_only_auroc(
    id_features: np.ndarray,
    ood_features: np.ndarray,
    score_column: int = 0,
    higher_is_ood: bool = True,
) -> float:
    """Compute AUROC using a single count-only feature as the OOD score.

    The score direction is controlled by the explicit ``higher_is_ood`` parameter:
    - ``higher_is_ood=True`` (default): Numerically larger feature values indicate
      higher OOD likelihood (score = +feature_value). This adheres strictly to the
      canonical evaluation framework convention: higher score = more OOD. For
      molecular novelty benchmarks such as SIDER, where out-of-distribution molecules
      are systematically larger on average than in-distribution molecules, this
      standard orientation correctly models size-based detection.
    - ``higher_is_ood=False``: Numerically smaller feature values indicate higher
      OOD likelihood (score = -feature_value).

    Score orientation MUST be set explicitly by benchmark hypothesis/semantics.
    It MUST NEVER be inferred post-hoc from test labels to artificially maximize AUROC.

    Parameters
    ----------
    id_features : np.ndarray
        Count features for in-distribution graphs, shape ``(n_id, 21)``.
    ood_features : np.ndarray
        Count features for OOD graphs, shape ``(n_ood, 21)``.
    score_column : int, default=0
        Which column of the feature matrix to use as the score (default 0 = n_atoms).
    higher_is_ood : bool, default=True
        Whether higher feature values represent higher OOD likelihood.
        - True: score = +feature_value (larger = more OOD, canonical default).
        - False: score = -feature_value (smaller = more OOD).

    Returns
    -------
    float
        AUROC in [0, 1].

    Raises
    ------
    ValueError
        If either array is empty.
    """
    if len(id_features) == 0 or len(ood_features) == 0:
        raise ValueError("Both id_features and ood_features must be non-empty.")

    # Apply explicit score direction
    sign = 1.0 if higher_is_ood else -1.0
    id_scores = sign * id_features[:, score_column]
    ood_scores = sign * ood_features[:, score_column]

    y_true = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])
    y_score = np.concatenate([id_scores, ood_scores])

    # Import from our own metrics module to avoid circular issues.
    from evaluation.metrics import compute_auroc
    return compute_auroc(y_true, y_score)


def filter_matched_controls(
    controls_df: pd.DataFrame,
    target_set: Optional[str] = None,
) -> pd.DataFrame:
    """Filter matched_controls.csv to successfully matched rows.

    Parameters
    ----------
    controls_df : pd.DataFrame
        Loaded from ``processed_v3/tables/matched_controls.csv``.
        Expected columns: target_uid, target_set, control_uid, control_set,
        d_atoms, d_bonds, matched.
    target_set : str, optional
        If given, also filter to rows where ``target_set == target_set``.

    Returns
    -------
    pd.DataFrame
        Rows where ``matched == True``, optionally filtered by target_set.
    """
    required_cols = {"target_uid", "target_set", "control_uid", "matched"}
    missing = required_cols - set(controls_df.columns)
    if missing:
        raise ValueError(
            f"matched_controls.csv missing columns: {missing}. "
            f"Available: {list(controls_df.columns)}"
        )
    df = controls_df[controls_df["matched"] == True].copy()
    if target_set is not None:
        df = df[df["target_set"] == target_set]
    return df


def load_eligible_pairs(
    pairs_path: str,
    split: Optional[str] = None,
    family: Optional[str] = None,
) -> pd.DataFrame:
    """Load and optionally filter eligible_pairs.csv.

    Parameters
    ----------
    pairs_path : str
        Path to ``processed_v3/tables/eligible_pairs.csv``.
    split : str, optional
        Filter to rows matching this split (e.g. "train", "val", "test").
    family : str, optional
        Filter to rows matching this corruption family (e.g. "A", "B", "C").

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with columns:
        corrupt_uid, twin_uid, split, family, n_edits, n_viol_atoms.
    """
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"eligible_pairs.csv not found: {pairs_path}")

    df = pd.read_csv(pairs_path)
    required = {"corrupt_uid", "twin_uid", "split", "family"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"eligible_pairs.csv missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )
    if split is not None:
        df = df[df["split"] == split]
    if family is not None:
        df = df[df["family"] == family]
    return df


# Valid benchmark and split names for processed_v3.
VALID_BENCHMARKS = frozenset([
    "validity/val/A", "validity/val/B",
    "validity/test/A", "validity/test/B", "validity/test/C",
    "novelty/val/novel_scaffold", "novelty/val/novel_dataset",
    "novelty/test/novel_scaffold", "novelty/test/novel_dataset",
])

VALID_SPLIT_NAMES = frozenset([
    "id_train", "id_val", "id_test",
    "corrupt_train", "corrupt_val", "corrupt_test",
    "novel_scaffold_val", "novel_scaffold_test",
    "novel_dataset_val", "novel_dataset_test",
])


def validate_split_name(name: str) -> bool:
    """Check if a split name is valid for processed_v3."""
    return name in VALID_SPLIT_NAMES


def validate_benchmark_name(name: str) -> bool:
    """Check if a benchmark name is valid for processed_v3."""
    return name in VALID_BENCHMARKS
