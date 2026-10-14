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

REQUIRED_COLUMNS = [
    "sample_id",
    "model_name",
    "benchmark",
    "true_label",
    "ood_score",
    "seed",
]


def validate_prediction_dataframe(
    df: pd.DataFrame,
    source_name: str = "DataFrame",
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a prediction DataFrame against the evaluator schema and constraints.

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
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Non-empty check
    if df.empty:
        errors.append(f"Submission '{source_name}' contains 0 data rows.")
        return False, errors, warnings

    # 2. Required columns check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(
            f"Missing required column(s): {missing_cols}. "
            f"Expected schema: {REQUIRED_COLUMNS}. Available columns: {list(df.columns)}"
        )
        # Cannot safely perform further column-specific validation
        return False, errors, warnings

    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra_cols:
        warnings.append(
            f"Found extra column(s) not in canonical schema: {extra_cols}. "
            "These will be ignored by evaluation/evaluate.py."
        )

    # 3. Check for null / empty values in required columns
    for col in REQUIRED_COLUMNS:
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
    if not df["true_label"].isna().all():
        raw_labels = df["true_label"].dropna()
        invalid_label_rows = []
        for idx, val in raw_labels.items():
            try:
                # Check if it parses as integer equal to 0 or 1
                val_float = float(val)
                if val_float not in (0.0, 1.0) or val_float != int(val_float):
                    invalid_label_rows.append((idx, val))
            except (ValueError, TypeError):
                invalid_label_rows.append((idx, val))

        if invalid_label_rows:
            bad_samples = invalid_label_rows[:10]
            count = len(invalid_label_rows)
            errors.append(
                f"Column 'true_label' contains {count} invalid value(s). "
                f"Must be binary integer 0 (ID) or 1 (OOD). First offenses: {bad_samples}"
                + (f" ... ({count - 10} more)" if count > 10 else "")
            )

    # 5. Validate ood_score (numeric, non-NaN, finite)
    if not df["ood_score"].isna().all():
        raw_scores = df["ood_score"].dropna()
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
                f"Column 'ood_score' contains {count} non-numeric or NaN value(s) at row(s): {non_numeric_rows[:10]}"
            )
        if infinite_rows:
            count = len(infinite_rows)
            errors.append(
                f"Column 'ood_score' contains {count} infinite (+/-inf) value(s) at row(s): {infinite_rows[:10]}"
            )

    # 6. Validate seed (integer)
    if not df["seed"].isna().all():
        raw_seeds = df["seed"].dropna()
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
                f"Column 'seed' contains {count} invalid seed value(s). Must be integer. "
                f"First offenses: {invalid_seed_rows[:10]}"
            )

    # 7. Check string fields for whitespace / empty strings
    for col in ["sample_id", "model_name", "benchmark"]:
        if col in df.columns:
            str_series = df[col].astype(str)
            empty_str_mask = str_series.str.strip() == ""
            if empty_str_mask.any():
                bad_idxs = df.index[empty_str_mask].tolist()[:10]
                errors.append(
                    f"Column '{col}' contains blank/whitespace-only string(s) at row index(es): {bad_idxs}"
                )

    # 8. Check class diversity within each (model_name, benchmark, seed) group
    # Only if required columns exist without catastrophic nulls
    if not errors:
        group_cols = ["model_name", "benchmark", "seed"]
        grouped = df.groupby(group_cols, sort=False)

        single_class_groups = []
        for group_keys, group_df in grouped:
            unique_classes = set(group_df["true_label"].unique())
            # We require both 0 and 1
            if not ({0, 1}.issubset(unique_classes)):
                single_class_groups.append((group_keys, list(unique_classes), len(group_df)))

        if single_class_groups:
            for g_keys, u_cls, count in single_class_groups:
                errors.append(
                    f"Evaluation group (model='{g_keys[0]}', benchmark='{g_keys[1]}', seed={g_keys[2]}) "
                    f"with {count} rows contains only class(es): {u_cls}. "
                    "Each evaluation group MUST contain both ID (0) and OOD (1) samples to calculate AUROC, AUPR, and FPR95."
                )

    # 9. Orientation advisory warning
    warnings.append(
        "SCORE ORIENTATION ADVISORY: The evaluation framework strictly requires that LARGER 'ood_score' "
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
