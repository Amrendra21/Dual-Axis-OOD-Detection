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


CANDIDATE_BENCHMARK_COLS = ["benchmark", "axis", "split", "task", "dataset"]
CANDIDATE_LABEL_COLS = ["label", "y_true", "true_label", "target", "is_ood"]
CANDIDATE_SCORE_COLS = ["ood_score", "y_score", "score", "anomaly_score", "pred_score"]
CANDIDATE_MODEL_COLS = ["model_name", "model", "method", "architecture"]
CANDIDATE_SEED_COLS = ["seed", "random_seed"]


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

    Each benchmark is evaluated independently.
    Missing results are never fabricated.
    """
    # Auto-detect column names if not explicitly specified
    label_col = label_col or detect_column(df, CANDIDATE_LABEL_COLS, required=True, col_type="label")
    score_col = score_col or detect_column(df, CANDIDATE_SCORE_COLS, required=True, col_type="score")
    benchmark_col = benchmark_col or detect_column(df, CANDIDATE_BENCHMARK_COLS, required=False, col_type="benchmark")
    model_col = model_col or detect_column(df, CANDIDATE_MODEL_COLS, required=False, col_type="model")
    seed_col = seed_col or detect_column(df, CANDIDATE_SEED_COLS, required=False, col_type="seed")

    # Strict check for NaNs in evaluation columns - do not fabricate replacement values
    eval_cols = [col for col in [label_col, score_col, benchmark_col, model_col, seed_col] if col]
    nan_counts = df[eval_cols].isna().sum()
    if (nan_counts > 0).any():
        cols_with_nans = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(
            f"Input DataFrame contains NaN values in evaluation columns: {cols_with_nans}. "
            "To prevent fabricated metrics, all evaluation rows must be complete. "
            "Please clean or filter your predictions before evaluation."
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
        y_true = df[label_col].to_numpy()
        y_score = df[score_col].to_numpy()
        metrics = compute_all_metrics(y_true, y_score)
        records.append({
            "model_name": "Unknown",
            "seed": "None",
            "benchmark": "Overall",
            **metrics,
        })
    else:
        grouped = df.groupby(group_cols, sort=False)
        for group_keys, group_df in grouped:
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            group_info = dict(zip(group_cols, group_keys))

            m_name = str(group_info.get(model_col, "Unknown"))
            s_val = str(group_info.get(seed_col, "None"))
            b_val = str(group_info.get(benchmark_col, "Overall"))

            y_true = group_df[label_col].to_numpy()
            y_score = group_df[score_col].to_numpy()

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
                **metrics,
            })

    results_df = pd.DataFrame(records)
    if results_df.empty:
        return results_df

    # Calculate combined dual-axis metrics ONLY when both axes are present.
    # Axis 1 = validity (corruption detection); Axis 2 = novelty (semantic OOD).
    #
    # processed_v3 benchmark names:
    #   Axis 1: validity/{val,test}/{A,B,C}
    #   Axis 2: novelty/{val,test}/{novel_scaffold,novel_dataset}
    #
    # Legacy names (still accepted for backward compatibility):
    #   Axis 1: anything containing "axis1", "structural", or "corrupt"
    #   Axis 2: anything containing "axis2", "semantic", "scaffold",
    #           "novelty", or "novel_dataset"
    #
    # IMPORTANT: Families A, B, C are NEVER pooled; each is its own benchmark.
    # The combined metric averages over ALL validity and ALL novelty benchmarks
    # that are present for a given model+seed.  It is never fabricated when only
    # one axis is available.

    combined_records = []
    if "benchmark" in results_df.columns:
        models = results_df["model_name"].unique()
        seeds = results_df["seed"].unique()

        for m in models:
            for s in seeds:
                sub = results_df[
                    (results_df["model_name"] == m) & (results_df["seed"] == s)
                ].copy()
                benchmarks_lower = sub["benchmark"].str.lower().tolist()

                def _is_validity(b: str) -> bool:
                    """Return True if benchmark b is an Axis-1 (validity) benchmark."""
                    return (
                        b.startswith("validity/")
                        or "axis1" in b
                        or "structural" in b
                        or ("corrupt" in b and "novelty" not in b)
                    )

                def _is_novelty(b: str) -> bool:
                    """Return True if benchmark b is an Axis-2 (novelty) benchmark."""
                    return (
                        b.startswith("novelty/")
                        or "axis2" in b
                        or "semantic" in b
                        or "novel_scaffold" in b
                        or "novel_dataset" in b
                    )

                validity_benchmarks = [b for b in benchmarks_lower if _is_validity(b)]
                novelty_benchmarks = [b for b in benchmarks_lower if _is_novelty(b)]

                if not (validity_benchmarks and novelty_benchmarks):
                    # Both axes must be present; do not fabricate the missing one.
                    continue

                # Collect all matching rows for each axis.
                validity_rows = sub[
                    sub["benchmark"].str.lower().isin(set(validity_benchmarks))
                ]
                novelty_rows = sub[
                    sub["benchmark"].str.lower().isin(set(novelty_benchmarks))
                ]

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
