"""
Visualization Module for Dual-Axis OOD Detection.

Generates publication-quality figures from actual prediction and result files:
- ROC Curves (Single and Multi-benchmark comparisons)
- Precision-Recall Curves
- Metric Comparison Bar Charts (AUROC, AUPR, FPR95 across models and axes)

Strict Non-Fabrication Policy:
- Plots are generated strictly from provided result files.
- If data is absent or incomplete, plots clearly reflect the available data or raise
  informative errors rather than generating speculative visuals.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for non-interactive / headless environment rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.metrics import compute_auroc, compute_aupr, compute_fpr95, validate_inputs, _compute_roc_curve_numpy


# Publication styling configuration
STYLE_CONFIG = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}


def apply_plot_style() -> None:
    """Apply clean publication styling."""
    plt.rcParams.update(STYLE_CONFIG)


def get_roc_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract FPR and TPR arrays using scikit-learn or NumPy fallback."""
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        return fpr, tpr
    except ImportError:
        fpr, tpr, _ = _compute_roc_curve_numpy(y_true, y_score)
        return fpr, tpr


def plot_roc_curve(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list],
    output_path: Optional[str] = None,
    title: str = "Receiver Operating Characteristic (ROC)",
    benchmark_name: str = "OOD Detector",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot ROC curve for a single benchmark evaluation.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels (0=ID, 1=OOD).
    y_score : array-like
        Predicted anomaly scores (higher = more OOD).
    output_path : str, optional
        Path to save figure.
    title : str
        Figure title.
    benchmark_name : str
        Label used in legend.
    ax : plt.Axes, optional
        Existing matplotlib axes.

    Returns
    -------
    plt.Figure
    """
    apply_plot_style()
    y_true_arr, y_score_arr = validate_inputs(y_true, y_score)

    fpr, tpr = get_roc_curve_data(y_true_arr, y_score_arr)
    auroc = compute_auroc(y_true_arr, y_score_arr)
    fpr95 = compute_fpr95(y_true_arr, y_score_arr, target_tpr=0.95)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        created_fig = True
    else:
        fig = ax.figure

    label_str = f"{benchmark_name}\n(AUROC = {auroc:.3f}, FPR95 = {fpr95:.3f})"
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=label_str)
    ax.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.2, linestyle="--", label="Random Chance (0.500)")

    # Highlight 95% TPR threshold point
    idx_95 = np.where(tpr >= 0.95)[0]
    if len(idx_95) > 0:
        pt_idx = idx_95[0]
        ax.scatter([fpr[pt_idx]], [tpr[pt_idx]], color="#d62728", zorder=5, s=40)
        ax.annotate(
            f"FPR@95%TPR: {fpr[pt_idx]:.2f}",
            xy=(fpr[pt_idx], tpr[pt_idx]),
            xytext=(fpr[pt_idx] + 0.08, tpr[pt_idx] - 0.08),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1),
            fontsize=9,
            color="#d62728",
            fontweight="bold",
        )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=300)
        if created_fig:
            plt.close(fig)

    return fig


def plot_multi_roc_curves(
    curves_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[str] = None,
    title: str = "Dual-Axis OOD Detection: ROC Curves",
) -> plt.Figure:
    """
    Plot overlaid ROC curves for multiple benchmarks (e.g., Axis 1 vs Axis 2).

    Parameters
    ----------
    curves_dict : Dict[str, Tuple[y_true, y_score]]
        Dictionary mapping benchmark or model names to (y_true, y_score) tuples.
    output_path : str, optional
        Path to save figure.
    title : str
        Figure title.
    """
    apply_plot_style()
    if not curves_dict:
        raise ValueError("curves_dict is empty. Cannot generate multi-ROC plot.")

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (name, (y_t, y_s)) in enumerate(curves_dict.items()):
        y_true_arr, y_score_arr = validate_inputs(y_t, y_s)
        fpr, tpr = get_roc_curve_data(y_true_arr, y_score_arr)
        auroc = compute_auroc(y_true_arr, y_score_arr)
        fpr95 = compute_fpr95(y_true_arr, y_score_arr, target_tpr=0.95)

        color = colors[i % len(colors)]
        label_text = f"{name} (AUROC={auroc:.3f}, FPR95={fpr95:.3f})"
        ax.plot(fpr, tpr, lw=2, color=color, label=label_text)

    ax.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.2, linestyle="--", label="Chance (0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    return fig


def plot_metric_comparison(
    summary_data: Union[pd.DataFrame, str],
    metrics: Tuple[str, ...] = ("auroc", "aupr_out", "fpr95"),
    output_path: Optional[str] = None,
    title: str = "Benchmark Metric Comparison",
) -> plt.Figure:
    """
    Plot grouped bar chart comparing performance metrics across benchmarks.

    Parameters
    ----------
    summary_data : pd.DataFrame or str
        Evaluation summary DataFrame or path to CSV file.
    metrics : tuple of str
        Metrics to plot.
    output_path : str, optional
        Path to save figure.
    title : str
        Figure title.
    """
    apply_plot_style()
    if isinstance(summary_data, str):
        if not os.path.exists(summary_data):
            raise FileNotFoundError(f"Summary file not found: {summary_data}")
        df = pd.read_csv(summary_data)
    else:
        df = summary_data.copy()

    if df.empty:
        raise ValueError("Summary DataFrame is empty. Cannot generate comparison plot.")

    # Identify relevant columns
    benchmark_col = "benchmark" if "benchmark" in df.columns else df.columns[0]
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        raise ValueError(f"None of requested metrics {metrics} found in data columns: {list(df.columns)}")

    benchmarks = df[benchmark_col].unique()
    n_benchmarks = len(benchmarks)
    n_metrics = len(available_metrics)

    fig, ax = plt.subplots(figsize=(max(8, n_benchmarks * 2.5), 5.5))
    bar_width = 0.8 / n_metrics
    x = np.arange(n_benchmarks)

    palette = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]

    for i, metric in enumerate(available_metrics):
        metric_values = []
        for b in benchmarks:
            sub = df[df[benchmark_col] == b]
            val = sub[metric].mean()
            metric_values.append(val)

        offset = (i - (n_metrics - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            metric_values,
            bar_width,
            label=metric.upper(),
            color=palette[i % len(palette)],
            edgecolor="black",
            linewidth=0.6,
        )

        # Label bar heights
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Benchmark / Split")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b).replace("_", " ") for b in benchmarks], rotation=15, ha="right")
    ax.set_ylim([0, 1.15])
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots from prediction or results CSV."
    )
    parser.add_argument(
        "--summary", "-s", type=str, default=None,
        help="Path to evaluation summary CSV (output from evaluate.py)."
    )
    parser.add_argument(
        "--predictions", "-p", type=str, default=None,
        help="Path to raw predictions CSV."
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="evaluation/results/plots",
        help="Directory to save generated plots."
    )

    args = parser.parse_args()

    if not args.summary and not args.predictions:
        print("Please provide either --summary or --predictions file.")
        parser.print_help()
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.summary:
        summary_path = args.summary
        out_bar = os.path.join(args.output_dir, "metric_comparison.png")
        plot_metric_comparison(summary_path, output_path=out_bar)
        print(f"Saved metric comparison plot to: {out_bar}")

    if args.predictions:
        df = pd.read_csv(args.predictions)
        # Check if benchmark column exists to generate multi-ROC
        bench_col = None
        for cand in ["benchmark", "axis", "split"]:
            if cand in df.columns:
                bench_col = cand
                break

        label_col = None
        for cand in ["label", "true_label", "y_true"]:
            if cand in df.columns:
                label_col = cand
                break

        score_col = None
        for cand in ["ood_score", "score", "y_score"]:
            if cand in df.columns:
                score_col = cand
                break

        if label_col and score_col:
            if bench_col:
                curves = {}
                for bench_val, group in df.groupby(bench_col):
                    y_t = group[label_col].to_numpy()
                    y_s = group[score_col].to_numpy()
                    if len(np.unique(y_t)) >= 2:
                        curves[str(bench_val)] = (y_t, y_s)

                if curves:
                    out_roc = os.path.join(args.output_dir, "multi_roc_curves.png")
                    plot_multi_roc_curves(curves, output_path=out_roc)
                    print(f"Saved multi-ROC curve to: {out_roc}")
            else:
                y_t = df[label_col].to_numpy()
                y_s = df[score_col].to_numpy()
                out_roc = os.path.join(args.output_dir, "roc_curve.png")
                plot_roc_curve(y_t, y_s, output_path=out_roc)
                print(f"Saved ROC curve to: {out_roc}")


if __name__ == "__main__":
    main()
