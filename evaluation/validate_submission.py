"""
Command-Line Submission Validator for Dual-Axis OOD Detection Predictions.

Validates that prediction CSV files submitted for evaluation match the schema,
data types, and group diversity constraints required by evaluation/evaluate.py:
- Canonical headers: sample_id, model_name, benchmark, true_label, ood_score, seed.
- Nonempty, finite numerical anomaly scores.
- Binary ground truth labels (0=ID, 1=OOD).
- Integer experimental seeds.
- Both ID and OOD classes present in every (model_name, benchmark, seed) group.
- Explicit advisory on semantic score orientation (higher = more OOD).
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

# Add project root to path for reliable imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

REQUIRED_COLUMNS = [
    "sample_id",
    "model_name",
    "benchmark",
    "true_label",
    "ood_score",
    "seed",
]

# Accepted aliases for Member 4 documented schema
ID_ALIASES = ["sample_id", "uid"]
BENCHMARK_ALIASES = ["benchmark", "benchmark/split", "split"]
SCORE_COLUMNS = ["ood_score", "validity_score", "novelty_score"]
LABEL_ALIASES = ["true_label", "label", "y_true", "ground_truth"]


def validate_prediction_dataframe(
    df: pd.DataFrame,
    source_name: str = "DataFrame",
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a prediction DataFrame against the evaluator schema and constraints.

    Supports both Member 5 canonical schema (sample_id, model_name, benchmark,
    true_label, ood_score, seed) and Member 4 documented schema (uid, validity_score,
    novelty_score, benchmark, split). Preserves all legacy validation rules.

    Parameters
    ----------
    df : pd.DataFrame
        Input prediction DataFrame.
    source_name : str
        Name or path of input source for logging.

    Returns
    -------
    Tuple[bool, List[str], List[str]]
        (is_valid, list_of_errors, list_of_warnings)
    """
    from evaluation.evaluate import is_validity_benchmark, is_novelty_benchmark

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Non-empty check
    if df.empty:
        errors.append(f"Submission '{source_name}' contains 0 data rows.")
        return False, errors, warnings

    df = df.copy()

    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in df.columns}

    # 2. Schema detection and required column checks
    # Identify ID column (sample_id or uid)
    id_col = None
    for alias in ID_ALIASES:
        if alias in cols_lower:
            id_col = cols_lower[alias]
            break

    # Identify Label column
    label_col = (
        cols_lower.get("true_label")
        or cols_lower.get("label")
        or cols_lower.get("y_true")
        or cols_lower.get("ground_truth")
    )

    # Identify Benchmark column:
    # If both 'benchmark' and 'split' exist as distinct columns, combine them into _resolved_benchmark
    # matching the exact partition resolution in evaluation/evaluate.py.
    benchmark_col = None
    if "benchmark" in cols_lower and "split" in cols_lower and cols_lower["benchmark"] != cols_lower["split"]:
        b_orig = cols_lower["benchmark"]
        s_orig = cols_lower["split"]
        def _combine(row):
            b_val = str(row[b_orig]).strip()
            s_val = str(row[s_orig]).strip()
            if "/" in b_val or s_val.lower() == "none" or not s_val:
                return b_val
            if "/" in s_val and s_val.lower().startswith(b_val.lower() + "/"):
                return s_val
            return f"{b_val}/{s_val}"
        df["_resolved_benchmark"] = df.apply(_combine, axis=1)
        benchmark_col = "_resolved_benchmark"
    else:
        for alias in BENCHMARK_ALIASES:
            if alias in cols_lower:
                benchmark_col = cols_lower[alias]
                break

    # Identify score columns present
    present_score_cols = [cols_lower[sc] for sc in SCORE_COLUMNS if sc in cols_lower]

    # Check for Member 4 vs Member 5 schema markers
    is_member4_schema = (
        ("uid" in cols_lower)
        or ("validity_score" in cols_lower)
        or ("novelty_score" in cols_lower)
        or ("ground_truth" in cols_lower)
    )

    # Missing critical columns
    if id_col is None:
        errors.append(
            f"Missing required identifier column: expected 'sample_id' (Member 5) or 'uid' (Member 4). Available columns: {cols}"
        )

    if label_col is None:
        errors.append(
            f"Missing required label column: expected 'true_label'. Available columns: {cols}"
        )

    if benchmark_col is None:
        errors.append(
            f"Missing required benchmark column: expected 'benchmark' or 'split'. Available columns: {cols}"
        )

    if not present_score_cols:
        errors.append(
            f"Missing required column(s): expected 'ood_score' (Member 5 schema) or 'validity_score'/'novelty_score' (Member 4 schema). "
            f"Expected schema: {REQUIRED_COLUMNS}. Available columns: {cols}"
        )

    # In Member 5 schema, model_name and seed are strictly required.
    # In Member 4 schema, they are optional with warnings.
    if "model_name" not in cols_lower and "model" not in cols_lower:
        if is_member4_schema:
            warnings.append("Member 4 schema detected without 'model_name' column; defaulting to 'Unknown' during evaluation.")
        else:
            errors.append(
                f"Missing required column(s): ['model_name']. Expected schema: {REQUIRED_COLUMNS}. Available columns: {cols}"
            )

    if "seed" not in cols_lower and "random_seed" not in cols_lower:
        if is_member4_schema:
            warnings.append("Member 4 schema detected without 'seed' column; defaulting to 'None' during evaluation.")
        else:
            errors.append(
                f"Missing required column(s): ['seed']. Expected schema: {REQUIRED_COLUMNS}. Available columns: {cols}"
            )

    if errors:
        return False, errors, warnings

    # Extra columns check
    known_cols = {
        "sample_id", "uid", "model_name", "model", "benchmark", "benchmark/split",
        "split", "true_label", "label", "y_true", "ground_truth", "ood_score",
        "validity_score", "novelty_score", "seed", "random_seed", "_resolved_benchmark",
    }
    extra_cols = [c for c in cols if c.lower() not in known_cols]
    if extra_cols:
        warnings.append(
            f"Found extra column(s) not in canonical schema: {extra_cols}. "
            "These will be ignored by evaluation/evaluate.py."
        )

    # 3. Check for null / empty values in active evaluation columns
    active_cols_to_check = [c for c in [id_col, label_col] if c and c in df.columns]
    for alias in BENCHMARK_ALIASES:
        if alias in cols_lower and cols_lower[alias] in df.columns and cols_lower[alias] not in active_cols_to_check:
            active_cols_to_check.append(cols_lower[alias])
    active_cols_to_check.extend(present_score_cols)
    if "model_name" in cols_lower:
        active_cols_to_check.append(cols_lower["model_name"])
    if "seed" in cols_lower:
        active_cols_to_check.append(cols_lower["seed"])

    for col in active_cols_to_check:
        null_mask = df[col].isna()
        if null_mask.any():
            bad_indices = df.index[null_mask].tolist()
            capped = bad_indices[:10]
            count = len(bad_indices)
            errors.append(
                f"Column '{col}' contains {count} null/NaN value(s) at row index(es): {capped}"
                + (f" ... ({count - 10} more)" if count > 10 else "")
            )

    # 4. Validate true_label (binary integers 0 and 1)
    if not df[label_col].isna().all():
        raw_labels = df[label_col].dropna()
        invalid_label_rows = []
        for idx, val in raw_labels.items():
            try:
                val_float = float(val)
                if val_float not in (0.0, 1.0) or val_float != int(val_float):
                    invalid_label_rows.append((idx, val))
            except (ValueError, TypeError):
                invalid_label_rows.append((idx, val))

        if invalid_label_rows:
            bad_samples = invalid_label_rows[:10]
            count = len(invalid_label_rows)
            errors.append(
                f"Column '{label_col}' contains {count} invalid value(s). "
                f"Must be binary integer 0 (ID) or 1 (OOD). First offenses: {bad_samples}"
                + (f" ... ({count - 10} more)" if count > 10 else "")
            )

    # 5. Validate score columns (numeric, non-NaN, finite)
    for sc in present_score_cols:
        if not df[sc].isna().all():
            raw_scores = df[sc].dropna()
            non_numeric_rows = []
            infinite_rows = []

            for idx, val in raw_scores.items():
                try:
                    val_float = float(val)
                    if np.isinf(val_float):
                        infinite_rows.append((idx, val))
                    elif np.isnan(val_float):
                        non_numeric_rows.append((idx, val))
                except (ValueError, TypeError):
                    non_numeric_rows.append((idx, val))

            if non_numeric_rows:
                count = len(non_numeric_rows)
                errors.append(
                    f"Column '{sc}' contains {count} non-numeric or NaN value(s) at row(s): {non_numeric_rows[:10]}"
                )
            if infinite_rows:
                count = len(infinite_rows)
                errors.append(
                    f"Column '{sc}' contains {count} infinite (+/-inf) value(s) at row(s): {infinite_rows[:10]}"
                )

    # 6. Validate seed (integer) if present
    seed_col_found = cols_lower.get("seed") or cols_lower.get("random_seed")
    if seed_col_found and not df[seed_col_found].isna().all():
        raw_seeds = df[seed_col_found].dropna()
        invalid_seed_rows = []
        for idx, val in raw_seeds.items():
            try:
                val_float = float(val)
                if val_float != int(val_float):
                    invalid_seed_rows.append((idx, val))
            except (ValueError, TypeError):
                invalid_seed_rows.append((idx, val))

        if invalid_seed_rows:
            count = len(invalid_seed_rows)
            errors.append(
                f"Column '{seed_col_found}' contains {count} invalid seed value(s). Must be integer. "
                f"First offenses: {invalid_seed_rows[:10]}"
            )

    # 7. Check string fields for whitespace / empty strings
    str_cols_to_check = [c for c in [id_col, benchmark_col] if c and c in df.columns]
    if "benchmark" in cols_lower and cols_lower["benchmark"] in df.columns:
        str_cols_to_check.append(cols_lower["benchmark"])
    if "split" in cols_lower and cols_lower["split"] in df.columns:
        str_cols_to_check.append(cols_lower["split"])
    str_cols_to_check = list(dict.fromkeys(str_cols_to_check))

    for col in str_cols_to_check:
        str_series = df[col].astype(str)
        empty_str_mask = str_series.str.strip() == ""
        if empty_str_mask.any():
            bad_idxs = df.index[empty_str_mask].tolist()[:10]
            errors.append(
                f"Column '{col}' contains blank/whitespace-only string(s) at row index(es): {bad_idxs}"
            )

    # 8. Check explicit benchmark semantics for Member 4 multi-score files
    has_val_score = "validity_score" in cols_lower
    has_nov_score = "novelty_score" in cols_lower
    if has_val_score and has_nov_score:
        unique_benchmarks = df[benchmark_col].dropna().unique()
        for b in unique_benchmarks:
            b_str = str(b).strip()
            is_val = is_validity_benchmark(b_str)
            is_nov = is_novelty_benchmark(b_str)
            if (is_val and is_nov) or (not is_val and not is_nov):
                errors.append(
                    f"Ambiguous score column mapping for benchmark '{b_str}': DataFrame contains both "
                    f"'validity_score' and 'novelty_score', but benchmark '{b_str}' lacks explicit validity "
                    f"or novelty semantics."
                )
    elif has_val_score and not has_nov_score and "ood_score" not in cols_lower:
        unique_benchmarks = df[benchmark_col].dropna().unique()
        for b in unique_benchmarks:
            b_str = str(b).strip()
            if is_novelty_benchmark(b_str):
                errors.append(
                    f"Invalid score mapping for novelty benchmark '{b_str}': DataFrame contains 'validity_score' "
                    f"but lacks 'novelty_score'."
                )
    elif has_nov_score and not has_val_score and "ood_score" not in cols_lower:
        unique_benchmarks = df[benchmark_col].dropna().unique()
        for b in unique_benchmarks:
            b_str = str(b).strip()
            if is_validity_benchmark(b_str):
                errors.append(
                    f"Invalid score mapping for validity benchmark '{b_str}': DataFrame contains 'novelty_score' "
                    f"but lacks 'validity_score'."
                )

    # 9. Check class diversity within each evaluation group
    if not errors:
        m_col = cols_lower.get("model_name") or cols_lower.get("model")
        s_col = cols_lower.get("seed") or cols_lower.get("random_seed")
        group_cols = [c for c in [m_col, benchmark_col, s_col] if c and c in df.columns]
        if not group_cols:
            group_cols = [benchmark_col]

        grouped = df.groupby(group_cols, sort=False)
        single_class_groups = []
        for group_keys, group_df in grouped:
            raw_unique = group_df[label_col].unique()
            unique_classes = set(int(x) if hasattr(x, "item") else x for x in raw_unique)
            if not ({0, 1}.issubset(unique_classes)):
                single_class_groups.append((group_keys, sorted(list(unique_classes)), len(group_df)))

        if single_class_groups:
            for g_keys, u_cls, count in single_class_groups:
                errors.append(
                    f"Evaluation group {g_keys} with {count} rows contains only class(es): {u_cls}. "
                    "Each evaluation group MUST contain both ID (0) and OOD (1) samples to calculate AUROC, AUPR, and FPR95."
                )

    # 10. Orientation advisory warning
    warnings.append(
        "SCORE ORIENTATION ADVISORY: The evaluation framework strictly requires that LARGER score "
        "values denote MORE OOD-LIKE (anomalous) inputs. Semantic score direction cannot be inferred from "
        "numerical ranges alone and MUST be verified by the model author (Member 4). Ensure in-distribution "
        "confidence scores (e.g., MSP) are properly inverted (1 - MSP or -MSP)."
    )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def validate_submission_file(
    file_path: str,
    quiet: bool = False,
) -> bool:
    """
    Validate a submission CSV file from disk.

    Parameters
    ----------
    file_path : str
        Path to prediction CSV.
    quiet : bool, default=False
        Suppress non-essential console output.

    Returns
    -------
    bool
        True if submission is valid, False otherwise.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Submission file not found: {file_path}", file=sys.stderr)
        return False

    try:
        # Check for 0-byte file
        if os.path.getsize(file_path) == 0:
            print(f"[ERROR] Submission file '{file_path}' is completely empty (0 bytes).", file=sys.stderr)
            return False

        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse CSV file '{file_path}': {e}", file=sys.stderr)
        return False

    is_valid, errors, warnings = validate_prediction_dataframe(df, source_name=file_path)

    if not quiet:
        print("=" * 80)
        print(f"PRE-FLIGHT SUBMISSION VALIDATION: {file_path}")
        print("=" * 80)
        print(f"Total Rows: {len(df)}")
        if not df.empty and set(REQUIRED_COLUMNS).issubset(df.columns):
            print(f"Models: {df['model_name'].unique().tolist()}")
            print(f"Benchmarks: {df['benchmark'].unique().tolist()}")
            print(f"Seeds: {df['seed'].unique().tolist()}")
        print("-" * 80)

        for w in warnings:
            print(f"[WARNING] {w}\n")

        if errors:
            print(f"[FAILURE] Found {len(errors)} validation error(s):")
            for i, err in enumerate(errors, start=1):
                print(f"  {i}. {err}")
            print("-" * 80)
            print("[RESULT] VALIDATION FAILED. Predictions cannot be processed by evaluate.py.")
            print("=" * 80)
        else:
            print("[RESULT] VALIDATION PASSED. File is fully compatible with evaluation/evaluate.py.")
            print("=" * 80)

    return is_valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight validation for Member 4 prediction CSV submissions."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to prediction CSV file to validate."
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress informational output, displaying only failures."
    )

    args = parser.parse_args()

    success = validate_submission_file(args.input, quiet=args.quiet)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
