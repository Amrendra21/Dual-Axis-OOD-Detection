"""
Command-Line Evaluation Runner for Dual-Axis OOD Detection.

Reads prediction CSV files from actual model evaluations, computes standard
metrics (AUROC, AUPR-Out, AUPR-In, FPR95) separately for each evaluated benchmark,
and outputs formatted summaries and CSV reports.

Benchmark naming convention (processed_v3 dataset):
  Axis 1 (validity):  validity/{val,test}/{A,B,C}
  Axis 2 (novelty):   novelty/{val,test}/{novel_scaffold,novel_dataset}

Strict Non-Fabrication Policy:
- Evaluates only benchmarks and models physically present in the input file.
- Never imputes, fabricates, or estimates missing predictions, splits, or metrics.
- Combined dual-axis metrics are computed only when both Axis 1 and Axis 2 results
  are present in the evaluated predictions.
- Corruption families A, B, and C are NEVER pooled; each is evaluated independently.
- Family C has a known element-count shortcut (count-only AUROC 0.798 on test);
  it is always reported separately and never merged into the headline validity metric.
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

from evaluation.metrics import compute_all_metrics, validate_inputs


CANDIDATE_ID_COLS = ["sample_id", "uid", "id", "graph_id", "molecule_id"]
CANDIDATE_BENCHMARK_COLS = ["benchmark", "benchmark/split", "split", "axis", "task", "dataset"]
CANDIDATE_LABEL_COLS = ["label", "y_true", "true_label", "ground_truth", "target", "is_ood"]
CANDIDATE_SCORE_COLS = ["ood_score", "y_score", "score", "anomaly_score", "pred_score"]
CANDIDATE_MODEL_COLS = ["model_name", "model", "method", "architecture"]
CANDIDATE_SEED_COLS = ["seed", "random_seed"]


def is_validity_benchmark(b: str) -> bool:
    """Return True if benchmark name indicates an Axis-1 (structural validity) benchmark."""
    s = str(b).strip().lower()
    return (
        s.startswith("validity/")
        or s.startswith("validity_")
        or s == "validity"
        or "axis1" in s
        or "structural" in s
        or ("corrupt" in s and "novelty" not in s)
    )


def is_novelty_benchmark(b: str) -> bool:
    """Return True if benchmark name indicates an Axis-2 (semantic novelty) benchmark."""
    s = str(b).strip().lower()
    return (
        s.startswith("novelty/")
        or s.startswith("novelty_")
        or s == "novelty"
        or "axis2" in s
        or "semantic" in s
        or "novel_scaffold" in s
        or "novel_dataset" in s
        or ("scaffold" in s and "corrupt" not in s)
    )


def detect_column(df: pd.DataFrame, candidates: List[str], required: bool = True, col_type: str = "") -> Optional[str]:
    """Auto-detect column name from candidates case-insensitively."""
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        available = list(df.columns)
        raise ValueError(
            f"Could not find required {col_type} column. "
            f"Expected one of {candidates}, but found: {available}"
        )
    return None


def generate_example_predictions_csv(output_path: str) -> None:
    """Generate an illustrative sample predictions CSV file for documentation.

    Demonstrates the processed_v3 benchmark naming convention:
      validity/{val,test}/{A,B,C}  and  novelty/{val,test}/{novel_scaffold,novel_dataset}

    SYNTHETIC TEST FIXTURE ONLY — scores are random; do not treat as research results.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # Each benchmark needs at least one ID (label=0) and one OOD (label=1) row.
    # true_label comes from split membership, NOT from the molecular task-label tensor.
    sample_data = {
        "sample_id": [
            # validity/test/A: id_test=0, corrupt_test family A=1
            "T21-00001", "T21-00002", "COR-A-T21-00001",
            # validity/test/B: id_test=0, corrupt_test family B=1
            "T21-00001", "T21-00002", "COR-B-T21-00002",
            # validity/test/C: id_test=0, corrupt_test family C=1 (held-out; report separately)
            "T21-00001", "T21-00002", "COR-C-T21-00003",
            # novelty/test/novel_scaffold: id_test=0, novel_scaffold_test=1
            "T21-00003", "T21-00004", "T21-SCF-00010",
            # novelty/test/novel_dataset: id_test=0, novel_dataset_test=1 (SIDER; y=NaN)
            "T21-00003", "T21-00004", "SID-00001",
        ],
        "model_name": ["GCN_Energy"] * 15,
        "benchmark": (
            ["validity/test/A"] * 3
            + ["validity/test/B"] * 3
            + ["validity/test/C"] * 3
            + ["novelty/test/novel_scaffold"] * 3
            + ["novelty/test/novel_dataset"] * 3
        ),
        "true_label": [0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1],
        "ood_score":  [0.12, 0.18, 0.87,  0.15, 0.22, 0.79,
                       0.11, 0.19, 0.93,  0.14, 0.25, 0.68,
                       0.13, 0.21, 0.71],
        "seed": [42] * 15,
    }
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    print(f"Sample template predictions written to: {output_path}")


def evaluate_dataframe(
    df: pd.DataFrame,
    benchmark_col: Optional[str] = None,
    label_col: Optional[str] = None,
    score_col: Optional[str] = None,
    model_col: Optional[str] = None,
    seed_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Evaluate OOD predictions contained in a DataFrame.

    Supports both Member 5 schema (sample_id, model_name, benchmark, true_label, ood_score, seed)
    and Member 4 schema (uid, validity_score, novelty_score, benchmark, split, etc.).
    Each benchmark is evaluated independently. Missing results are never fabricated.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Auto-detect label, model, and seed columns
    label_col = label_col or detect_column(df, CANDIDATE_LABEL_COLS, required=True, col_type="label")
    model_col = model_col or detect_column(df, CANDIDATE_MODEL_COLS, required=False, col_type="model")
    seed_col = seed_col or detect_column(df, CANDIDATE_SEED_COLS, required=False, col_type="seed")

    # Benchmark resolution:
    # 1. If benchmark_col explicitly given, use it.
    # 2. If both 'benchmark' and 'split' exist as distinct columns, combine them if benchmark is generic.
    # 3. Else detect from CANDIDATE_BENCHMARK_COLS.
    if benchmark_col is None:
        cols_lower = {c.lower(): c for c in df.columns}
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
            benchmark_col = detect_column(df, CANDIDATE_BENCHMARK_COLS, required=False, col_type="benchmark")

    # Detect score columns present in DataFrame
    cols_lower = {c.lower(): c for c in df.columns}
    has_validity_col = "validity_score" in cols_lower
    has_novelty_col = "novelty_score" in cols_lower
    validity_col_name = cols_lower.get("validity_score")
    novelty_col_name = cols_lower.get("novelty_score")

    detected_legacy_score = detect_column(df, CANDIDATE_SCORE_COLS, required=False, col_type="score")

    # Validate explicit score_col if supplied
    if score_col is not None:
        if score_col not in df.columns:
            if score_col.lower() in cols_lower:
                score_col = cols_lower[score_col.lower()]
            else:
                raise ValueError(f"Specified score_col '{score_col}' not found in DataFrame columns: {list(df.columns)}")
    else:
        # If score_col is None, verify that at least one valid score candidate exists
        if not (has_validity_col or has_novelty_col or detected_legacy_score):
            raise ValueError(
                f"Could not find required score column. "
                f"Expected one of {CANDIDATE_SCORE_COLS}, 'validity_score', or 'novelty_score', "
                f"but found: {list(df.columns)}"
            )

    # Score column selector per benchmark group
    def _get_score_col_for_benchmark(b_name: str) -> str:
        if score_col is not None:
            return score_col

        is_val = is_validity_benchmark(b_name)
        is_nov = is_novelty_benchmark(b_name)

        if is_val and is_nov:
            raise ValueError(
                f"Ambiguous benchmark semantics for '{b_name}': matches both validity and novelty criteria. "
                f"Cannot select score column automatically. Specify score_col explicitly."
            )

        if is_val:
            if has_validity_col:
                return validity_col_name
            elif detected_legacy_score:
                return detected_legacy_score
            else:
                raise ValueError(
                    f"Invalid score mapping for validity benchmark '{b_name}': DataFrame contains 'novelty_score' "
                    f"but lacks 'validity_score' (or legacy score column). Cannot score validity with novelty scores."
                )

        if is_nov:
            if has_novelty_col:
                return novelty_col_name
            elif detected_legacy_score:
                return detected_legacy_score
            else:
                raise ValueError(
                    f"Invalid score mapping for novelty benchmark '{b_name}': DataFrame contains 'validity_score' "
                    f"but lacks 'novelty_score' (or legacy score column). Cannot score novelty with validity scores."
                )

        # Non-axis specific benchmark (e.g. "Overall", "custom_task")
        if has_validity_col and has_novelty_col:
            raise ValueError(
                f"Ambiguous score column selection for benchmark '{b_name}': DataFrame contains both "
                f"'{validity_col_name}' and '{novelty_col_name}', but benchmark semantics are not explicitly "
                f"validity or novelty. Please specify score_col explicitly."
            )
        elif has_validity_col and not detected_legacy_score:
            return validity_col_name
        elif has_novelty_col and not detected_legacy_score:
            return novelty_col_name
        elif detected_legacy_score and not has_validity_col and not has_novelty_col:
            return detected_legacy_score
        else:
            candidates_present = [c for c in [validity_col_name, novelty_col_name, detected_legacy_score] if c]
            raise ValueError(
                f"Ambiguous score column selection for benchmark '{b_name}': Multiple candidate score columns "
                f"present ({candidates_present}). Please specify score_col explicitly."
            )

    # Determine grouping columns
    group_cols = []
    if model_col:
        group_cols.append(model_col)
    if seed_col:
        group_cols.append(seed_col)
    if benchmark_col:
        group_cols.append(benchmark_col)

    records = []

    if not group_cols:
        # Single benchmark, single model
        b_val = "Overall"
        chosen_score_col = _get_score_col_for_benchmark(b_val)
        if df[label_col].isna().any() or df[chosen_score_col].isna().any():
            raise ValueError(f"Input DataFrame contains NaN values in evaluation columns.")
        y_true = df[label_col].to_numpy()
        y_score = df[chosen_score_col].to_numpy()
        metrics = compute_all_metrics(y_true, y_score)
        records.append({
            "model_name": "Unknown",
            "seed": "None",
            "benchmark": b_val,
            "score_col": chosen_score_col,
            **metrics,
        })
    else:
        # Pre-flight check: ensure label and grouping columns have no NaNs
        for col in [label_col] + group_cols:
            if df[col].isna().any():
                raise ValueError(f"Input DataFrame contains NaN values in evaluation column '{col}'.")

        grouped = df.groupby(group_cols, sort=False)
        for group_keys, group_df in grouped:
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            group_info = dict(zip(group_cols, group_keys))

            m_name = str(group_info.get(model_col, "Unknown"))
            s_val = str(group_info.get(seed_col, "None"))
            b_val = str(group_info.get(benchmark_col, "Overall"))

            chosen_score_col = _get_score_col_for_benchmark(b_val)

            # Check NaNs in the chosen score column for this partition
            if group_df[chosen_score_col].isna().any():
                raise ValueError(
                    f"Input DataFrame contains NaN values in '{chosen_score_col}' for benchmark '{b_val}'."
                )

            y_true = group_df[label_col].to_numpy()
            y_score = group_df[chosen_score_col].to_numpy()

            # Verify that both classes exist in this partition
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                print(
                    f"[WARNING] Skipping benchmark '{b_val}' for model '{m_name}' (seed {s_val}): "
                    f"Only class {unique_classes.tolist()} is present. Missing results are not fabricated."
                )
                continue

            metrics = compute_all_metrics(y_true, y_score)
            records.append({
                "model_name": m_name,
                "seed": s_val,
                "benchmark": b_val,
                "score_col": chosen_score_col,
                **metrics,
            })

    results_df = pd.DataFrame(records)
    if results_df.empty:
        return results_df

    # Calculate combined dual-axis metrics ONLY when both axes are present.
    combined_records = []
    if "benchmark" in results_df.columns:
        models = results_df["model_name"].unique()
        seeds = results_df["seed"].unique()

        for m in models:
            for s in seeds:
                sub = results_df[
                    (results_df["model_name"] == m) & (results_df["seed"] == s)
                ].copy()
                benchmarks_list = sub["benchmark"].tolist()

                validity_benchmarks = [b for b in benchmarks_list if is_validity_benchmark(b)]
                novelty_benchmarks = [b for b in benchmarks_list if is_novelty_benchmark(b)]

                if not (validity_benchmarks and novelty_benchmarks):
                    # Both axes must be present; do not fabricate the missing one.
                    continue

                # Collect all matching rows for each axis.
                validity_rows = sub[sub["benchmark"].isin(set(validity_benchmarks))]
                novelty_rows = sub[sub["benchmark"].isin(set(novelty_benchmarks))]

                # Average metrics across all validity benchmarks, then across all
                # novelty benchmarks, then combine. This preserves per-family
                # separation in the individual rows while still providing a
                # useful combined summary.
                metric_cols = ["auroc", "aupr_out", "aupr_in", "fpr95"]
                val_means = validity_rows[metric_cols].mean()
                nov_means = novelty_rows[metric_cols].mean()

                mean_auroc = float((val_means["auroc"] + nov_means["auroc"]) / 2.0)
                if val_means["auroc"] > 0 and nov_means["auroc"] > 0:
                    harmonic_auroc = float(
                        2.0 / (1.0 / val_means["auroc"] + 1.0 / nov_means["auroc"])
                    )
                else:
                    harmonic_auroc = 0.0

                mean_aupr = float((val_means["aupr_out"] + nov_means["aupr_out"]) / 2.0)
                mean_fpr95 = float((val_means["fpr95"] + nov_means["fpr95"]) / 2.0)

                combined_records.append({
                    "model_name": m,
                    "seed": s,
                    "benchmark": "combined_dual_axis",
                    "auroc": round(mean_auroc, 4),
                    "aupr_out": round(mean_aupr, 4),
                    "aupr_in": round(
                        float((val_means["aupr_in"] + nov_means["aupr_in"]) / 2.0), 4
                    ),
                    "fpr95": round(mean_fpr95, 4),
                    "fpr95_percent": round(mean_fpr95 * 100.0, 2),
                    "n_id": int(validity_rows["n_id"].sum() + novelty_rows["n_id"].sum()),
                    "n_ood": int(validity_rows["n_ood"].sum() + novelty_rows["n_ood"].sum()),
                    "n_total": int(
                        validity_rows["n_total"].sum() + novelty_rows["n_total"].sum()
                    ),
                    "harmonic_mean_auroc": round(harmonic_auroc, 4),
                    "n_validity_benchmarks": len(validity_benchmarks),
                    "n_novelty_benchmarks": len(novelty_benchmarks),
                })

    if combined_records:
        combined_df = pd.DataFrame(combined_records)
        results_df = pd.concat([results_df, combined_df], ignore_index=True)

    return results_df


def run_evaluation(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    benchmark_col: Optional[str] = None,
    label_col: Optional[str] = None,
    score_col: Optional[str] = None,
    model_col: Optional[str] = None,
    seed_col: Optional[str] = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """Read prediction CSV and execute evaluation pipeline."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input predictions file not found: {input_path}")

    df = pd.read_csv(input_path)
    if not quiet:
        print(f"Loaded {len(df)} predictions from: {input_path}")

    results = evaluate_dataframe(
        df=df,
        benchmark_col=benchmark_col,
        label_col=label_col,
        score_col=score_col,
        model_col=model_col,
        seed_col=seed_col,
    )

    if results.empty:
        if not quiet:
            print("[WARNING] No valid evaluation benchmarks could be computed.")
        return results

    if not quiet:
        print("\n" + "=" * 80)
        print("DUAL-AXIS OOD DETECTION EVALUATION SUMMARY")
        print("=" * 80)
        display_cols = [c for c in ["model_name", "seed", "benchmark", "auroc", "aupr_out", "fpr95", "n_id", "n_ood"] if c in results.columns]
        print(results[display_cols].to_string(index=False))
        print("=" * 80 + "\n")

    # Determine output path
    if output_path is None:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "evaluation_summary.csv")
        else:
            default_dir = os.path.join(PROJECT_ROOT, "evaluation", "results")
            os.makedirs(default_dir, exist_ok=True)
            output_path = os.path.join(default_dir, "evaluation_summary.csv")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    results.to_csv(output_path, index=False)
    if not quiet:
        print(f"Evaluation results successfully saved to: {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Command-Line Evaluation Runner for Dual-Axis OOD Detection."
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to prediction CSV file containing labels and OOD scores."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to write evaluation results CSV."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save evaluation summary results."
    )
    parser.add_argument(
        "--benchmark-col", type=str, default=None,
        help="Column name denoting benchmark / axis / split."
    )
    parser.add_argument(
        "--label-col", type=str, default=None,
        help="Column name denoting ground truth binary label (0=ID, 1=OOD)."
    )
    parser.add_argument(
        "--score-col", type=str, default=None,
        help="Column name denoting predicted anomaly score (larger = more OOD)."
    )
    parser.add_argument(
        "--model-col", type=str, default=None,
        help="Column name denoting model identifier."
    )
    parser.add_argument(
        "--seed-col", type=str, default=None,
        help="Column name denoting random seed."
    )
    parser.add_argument(
        "--create-example-csv", type=str, default=None,
        help="Create a template predictions CSV file at specified path and exit."
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress console output."
    )

    args = parser.parse_args()

    if args.create_example_csv:
        generate_example_predictions_csv(args.create_example_csv)
        sys.exit(0)

    if not args.input:
        parser.print_help()
        sys.exit(1)

    try:
        run_evaluation(
            input_path=args.input,
            output_path=args.output,
            output_dir=args.output_dir,
            benchmark_col=args.benchmark_col,
            label_col=args.label_col,
            score_col=args.score_col,
            model_col=args.model_col,
            seed_col=args.seed_col,
            quiet=args.quiet,
        )
    except Exception as e:
        print(f"Evaluation Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
