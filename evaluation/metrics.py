"""
Evaluation Metrics for Dual-Axis Out-of-Distribution (OOD) Detection.

This module provides standard, robust evaluation metrics for OOD detection:
- AUROC: Area Under the Receiver Operating Characteristic Curve.
- AUPR (AUPR-Out / AUPR-In): Area Under the Precision-Recall Curve.
- FPR95: False Positive Rate at 95% True Positive Rate.

Conventions & Standards:
- Binary Labels: ID = 0 (negative / in-distribution), OOD = 1 (positive / anomaly).
- Score Orientation: Larger scores indicate higher anomaly/OOD likelihood.
- Pure NumPy & SciPy fallbacks are provided to ensure operation without strict
  scikit-learn dependency, with scikit-learn acceleration when installed.
"""

from typing import Dict, Tuple, Union
# pyrefly: ignore [missing-import]
import numpy as np


def validate_inputs(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and sanitize input ground-truth labels and anomaly scores.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 for ID, 1 for OOD).
    y_score : array-like
        Predicted OOD scores (higher = more OOD-like).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Validated 1D numpy arrays (y_true as int64, y_score as float64).

    Raises
    ------
    ValueError
        If inputs have mismatched lengths, contain NaNs/Infs, have non-binary
        labels, are empty, or contain only a single class.
    """
    # Check non-empty first
    if len(y_true) == 0 or len(y_score) == 0:
        raise ValueError("Input arrays must not be empty.")

    # Convert y_true to float64 array to validate numerical contents
    try:
        y_true_arr = np.asarray(y_true, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"y_true must contain numeric binary labels (0 for ID, 1 for OOD): {e}")

    # Convert y_score to float64 array
    try:
        y_score_arr = np.asarray(y_score, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"y_score must contain numeric anomaly scores: {e}")

    # Check dimensionality
    if y_true_arr.ndim != 1 or y_score_arr.ndim != 1:
        raise ValueError(
            f"Expected 1D arrays, got y_true.ndim={y_true_arr.ndim} and "
            f"y_score.ndim={y_score_arr.ndim}."
        )

    # Check mismatched lengths
    if len(y_true_arr) != len(y_score_arr):
        raise ValueError(
            f"Mismatched lengths: len(y_true)={len(y_true_arr)} != "
            f"len(y_score)={len(y_score_arr)}."
        )

    # Check for NaN / Inf in labels
    if np.isnan(y_true_arr).any() or np.isinf(y_true_arr).any():
        raise ValueError("y_true contains NaN or infinite values.")

    # Check for NaN / Inf in scores
    if np.isnan(y_score_arr).any() or np.isinf(y_score_arr).any():
        raise ValueError("y_score contains NaN or infinite values.")

    # Validate binary labels {0, 1}
    unique_vals = np.unique(y_true_arr)
    if not np.all(np.isin(unique_vals, [0.0, 1.0])):
        raise ValueError(
            f"Invalid labels in y_true: {unique_vals.tolist()}. "
            "Expected binary labels: 0 for ID and 1 for OOD."
        )

    y_true_int = y_true_arr.astype(np.int64)

    # Check class diversity (must contain both 0 and 1)
    if len(unique_vals) < 2:
        raise ValueError(
            f"y_true must contain both ID (0) and OOD (1) classes to evaluate OOD detection. "
            f"Found only class: {unique_vals.tolist()}"
        )

    return y_true_int, y_score_arr


def _compute_roc_curve_numpy(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve (FPR, TPR, thresholds) using NumPy.
    Scores are sorted descending.
    """
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    # Find distinct score threshold boundaries
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    # Cumulative true positives and false positives
    tps = np.cumsum(y_true_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_true_sorted == 0)[threshold_idxs]

    total_pos = tps[-1]
    total_neg = fps[-1]

    tpr = tps / total_pos
    fpr = fps / total_neg
    thresholds = y_score_sorted[threshold_idxs]

    # Prepend starting point (FPR=0, TPR=0)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[thresholds[0] + 1.0, thresholds]

    return fpr, tpr, thresholds


def compute_auroc(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list]
) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (AUROC).

    Measures discrimination between In-Distribution (0) and OOD (1).
    Larger scores indicate higher OOD likelihood.

    Parameters
    ----------
    y_true : array-like
        Binary labels (0=ID, 1=OOD).
    y_score : array-like
        Anomaly scores (higher = more OOD).

    Returns
    -------
    float
        AUROC score in [0.0, 1.0].
    """
    y_true_arr, y_score_arr = validate_inputs(y_true, y_score)

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true_arr, y_score_arr))
    except ImportError:
        # Exact Wilcoxon-Mann-Whitney U statistic calculation via SciPy
        try:
            # pyrefly: ignore [missing-import]
            from scipy.stats import mannwhitneyu
            pos_scores = y_score_arr[y_true_arr == 1]
            neg_scores = y_score_arr[y_true_arr == 0]
            u_stat, _ = mannwhitneyu(pos_scores, neg_scores, alternative="two-sided")
            return float(u_stat / (len(pos_scores) * len(neg_scores)))
        except ImportError:
            # Pure NumPy trapezoidal integration
            fpr, tpr, _ = _compute_roc_curve_numpy(y_true_arr, y_score_arr)
            # Use trapezoidal rule
            if hasattr(np, "trapezoid"):
                return float(np.trapezoid(tpr, fpr))
            return float(np.trapz(tpr, fpr))


def compute_aupr(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list],
    pos_label: int = 1
) -> float:
    """
    Compute Area Under the Precision-Recall Curve (AUPR / Average Precision).

    Parameters
    ----------
    y_true : array-like
        Binary labels (0=ID, 1=OOD).
    y_score : array-like
        Anomaly scores (higher = more OOD).
    pos_label : int, default=1
        Class to treat as positive.
        - pos_label=1: AUPR-Out (OOD considered positive class).
        - pos_label=0: AUPR-In (ID considered positive class).

    Returns
    -------
    float
        AUPR score in [0.0, 1.0].
    """
    y_true_arr, y_score_arr = validate_inputs(y_true, y_score)

    if pos_label not in (0, 1):
        raise ValueError(f"pos_label must be 0 or 1, got {pos_label}")

    # If ID is positive class, invert the scores so higher score means more ID-like
    adjusted_scores = y_score_arr if pos_label == 1 else -y_score_arr
    adjusted_labels = (y_true_arr == pos_label).astype(np.int64)

    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(adjusted_labels, adjusted_scores))
    except ImportError:
        # Pure NumPy Average Precision calculation
        desc_indices = np.argsort(adjusted_scores, kind="mergesort")[::-1]
        sorted_labels = adjusted_labels[desc_indices]

        tps = np.cumsum(sorted_labels == 1)
        fps = np.cumsum(sorted_labels == 0)
        total_pos = (adjusted_labels == 1).sum()

        precisions = tps / (tps + fps)
        recalls = tps / total_pos

        # Compute average precision as weighted sum of precisions by delta recall
        recalls_with_zero = np.r_[0.0, recalls]
        recall_diffs = np.diff(recalls_with_zero)
        ap = np.sum(precisions * recall_diffs)
        return float(np.clip(ap, 0.0, 1.0))


def compute_fpr95(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list],
    target_tpr: float = 0.95,
    as_percent: bool = False
) -> float:
    """
    Compute False Positive Rate at target True Positive Rate (default 95% TPR).

    Measures the fraction of In-Distribution (ID, label=0) examples incorrectly
    flagged as OOD when the detection threshold is set so that at least
    target_tpr (95%) of OOD (label=1) examples are correctly flagged.

    Parameters
    ----------
    y_true : array-like
        Binary labels (0=ID, 1=OOD).
    y_score : array-like
        Anomaly scores (higher = more OOD).
    target_tpr : float, default=0.95
        Desired minimum True Positive Rate (e.g. 0.95 for 95%).
    as_percent : bool, default=False
        If True, return in [0, 100]. If False, return in [0.0, 1.0].

    Returns
    -------
    float
        False positive rate at target TPR.
    """
    y_true_arr, y_score_arr = validate_inputs(y_true, y_score)

    if not (0.0 < target_tpr <= 1.0):
        raise ValueError(f"target_tpr must be in (0.0, 1.0], got {target_tpr}")

    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true_arr, y_score_arr, pos_label=1)
    except ImportError:
        fpr, tpr, thresholds = _compute_roc_curve_numpy(y_true_arr, y_score_arr)

    # Find the earliest index where TPR achieves or exceeds target_tpr
    qualifying_indices = np.where(tpr >= target_tpr)[0]
    if len(qualifying_indices) == 0:
        fpr_val = 1.0
    else:
        best_idx = qualifying_indices[0]
        fpr_val = float(fpr[best_idx])

    fpr_val = float(np.clip(fpr_val, 0.0, 1.0))
    return fpr_val * 100.0 if as_percent else fpr_val


def compute_all_metrics(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list]
) -> Dict[str, Union[float, int]]:
    """
    Compute all standard OOD detection metrics at once.

    Parameters
    ----------
    y_true : array-like
        Binary labels (0=ID, 1=OOD).
    y_score : array-like
        Anomaly scores (higher = more OOD).

    Returns
    -------
    Dict[str, Union[float, int]]
        Dictionary containing:
        - 'auroc': float in [0.0, 1.0]
        - 'aupr_out': float in [0.0, 1.0] (OOD positive)
        - 'aupr_in': float in [0.0, 1.0] (ID positive)
        - 'fpr95': float in [0.0, 1.0]
        - 'fpr95_percent': float in [0.0, 100.0]
        - 'n_id': int count of ID samples
        - 'n_ood': int count of OOD samples
        - 'n_total': int total samples
    """
    y_true_arr, y_score_arr = validate_inputs(y_true, y_score)

    n_id = int((y_true_arr == 0).sum())
    n_ood = int((y_true_arr == 1).sum())

    auroc = compute_auroc(y_true_arr, y_score_arr)
    aupr_out = compute_aupr(y_true_arr, y_score_arr, pos_label=1)
    aupr_in = compute_aupr(y_true_arr, y_score_arr, pos_label=0)
    fpr95 = compute_fpr95(y_true_arr, y_score_arr, target_tpr=0.95, as_percent=False)

    return {
        "auroc": auroc,
        "aupr_out": aupr_out,
        "aupr_in": aupr_in,
        "fpr95": fpr95,
        "fpr95_percent": fpr95 * 100.0,
        "n_id": n_id,
        "n_ood": n_ood,
        "n_total": len(y_true_arr),
    }
