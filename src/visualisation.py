"""
visualisation.py
----------------
Plotting and results-reporting utilities for the AdaptiveSAGE experiment.

Produces:
    Figure 1 – Training curves (loss, validation MRR, average depth).
    Figure 2 – Baseline comparison bar charts (MRR and Hits@20).
    Figure 3 – Computational efficiency trade-off (depth bar + scatter).
    Figure 4 – Early-exit layer distribution bar chart.
    LaTeX table – Copy-pasteable results table for the report.

All plot functions save to disk (dpi=300) and display inline.
"""

import matplotlib.pyplot as plt
import pandas as pd


# HeaRT Cora baseline numbers (Zhu et al., 2023)
HEART_BASELINES = {
    "Random":              {"mrr": 0.002, "hits@20": 0.01,  "depth": 0},
    "Common Neighbors":    {"mrr": 0.105, "hits@20": 0.30,  "depth": 0},
    "Resource Allocation": {"mrr": 0.132, "hits@20": 0.36,  "depth": 0},
    "GCN (HeaRT)":         {"mrr": 0.166, "hits@20": 0.42,  "depth": 3.0},
    "GraphSAGE (HeaRT)":   {"mrr": 0.161, "hits@20": 0.40,  "depth": 3.0},
}


# ---------------------------------------------------------------------------
# Figure 1: Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(results_log: dict, save_path: str = "training_curves.png"):
    """
    Plot a 3-panel training dynamics figure.

    Panels: training loss | validation MRR | average training depth.

    Args:
        results_log (dict): Experiment log with 'training_history' list of epoch dicts.
            Required keys per epoch: 'epoch', 'train_loss', 'val_mrr', 'train_avg_depth'.
        save_path (str): Output file path. Default: 'training_curves.png'.
    """
    df = pd.DataFrame(results_log["training_history"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(
        df["epoch"], df["train_loss"], "o-", linewidth=2, markersize=6, label="Training Loss"
    )
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        df["epoch"], df["val_mrr"] * 100, "s-", color="green",
        linewidth=2, markersize=6, label="Validation MRR"
    )
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("MRR (%)", fontsize=12)
    axes[1].set_title("Validation MRR", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        df["epoch"], df["train_avg_depth"], "^-", color="orange",
        linewidth=2, markersize=6, label="Training Depth"
    )
    axes[2].axhline(y=3.0, color="red", linestyle="--", alpha=0.5, label="Fixed Depth-3")
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Average Depth", fontsize=12)
    axes[2].set_title("Model Depth", fontsize=14, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 2: Baseline comparison
# ---------------------------------------------------------------------------

def plot_baseline_comparison(your_results: dict, save_path: str = "baseline_comparison.png"):
    """
    Bar chart comparing model MRR and Hits@20 against HeaRT baselines.

    Args:
        your_results (dict): Must contain 'mrr', 'hits@20', 'avg_depth'.
        save_path (str): Output file path. Default: 'baseline_comparison.png'.
    """
    baselines = dict(HEART_BASELINES)
    baselines["Ours (Adaptive)"] = {
        "mrr": your_results["mrr"],
        "hits@20": your_results.get("hits@20", 0.0),
        "depth": your_results.get("avg_depth", 2.0),
    }

    methods = list(baselines.keys())
    mrrs = [baselines[m]["mrr"] * 100 for m in methods]
    hits = [baselines[m]["hits@20"] * 100 for m in methods]
    colors = ["lightgray"] * (len(methods) - 1) + ["steelblue"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = axes[0].bar(range(len(methods)), mrrs, color=colors, edgecolor="black", linewidth=1.5)
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, rotation=45, ha="right")
    axes[0].set_ylabel("MRR (%)", fontsize=12)
    axes[0].set_title("Mean Reciprocal Rank Comparison", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, mrrs):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    bars2 = axes[1].bar(range(len(methods)), hits, color=colors, edgecolor="black", linewidth=1.5)
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(methods, rotation=45, ha="right")
    axes[1].set_ylabel("Hits@20 (%)", fontsize=12)
    axes[1].set_title("Hits@20 Comparison", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, hits):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: {save_path}")

    # Console table
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Method':<25} | {'MRR (%)':<10} | {'Hits@20 (%)':<12} | {'Avg Depth':<10}")
    print("-" * 80)
    for method in methods:
        mrr = baselines[method]["mrr"] * 100
        h20 = baselines[method]["hits@20"] * 100
        depth = baselines[method]["depth"]
        depth_str = f"{depth:.1f}" if depth > 0 else "N/A"
        print(f"{method:<25} | {mrr:<10.2f} | {h20:<12.2f} | {depth_str:<10}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Figure 3: Efficiency trade-off
# ---------------------------------------------------------------------------

def plot_efficiency_comparison(your_results: dict, save_path: str = "efficiency_comparison.png"):
    """
    Two-panel efficiency figure: depth bar chart + MRR vs computation-saved scatter.

    The x-axis label is 'Expected Depth Reduction (%)' to clarify that savings
    are theoretical (soft ACT aggregation, not hard early exit).

    Args:
        your_results (dict): Must contain 'mrr', 'avg_depth'.
        save_path (str): Output file path. Default: 'efficiency_comparison.png'.
    """
    methods = ["GCN\n(Fixed-3)", "GraphSAGE\n(Fixed-3)", "Ours\n(Adaptive)"]
    depths = [3.0, 3.0, your_results.get("avg_depth", 2.1)]
    mrrs = [16.6, 16.1, your_results["mrr"] * 100]
    depth_saved_pct = [(3.0 - d) / 3.0 * 100 for d in depths]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["lightcoral", "lightblue", "seagreen"]
    bars = axes[0].bar(methods, depths, color=colors, edgecolor="black", linewidth=1.5)
    axes[0].set_ylabel("Average Depth (Layers)", fontsize=12)
    axes[0].set_title("Computational Depth", fontsize=14, fontweight="bold")
    axes[0].set_ylim(0, 3.5)
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, depth in zip(bars, depths):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{depth:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    axes[1].scatter(
        depth_saved_pct[:2], mrrs[:2], s=200, color="lightgray",
        edgecolor="black", linewidth=2, label="Baselines", zorder=3
    )
    axes[1].scatter(
        depth_saved_pct[2], mrrs[2], s=300, color="steelblue",
        edgecolor="black", linewidth=2, label="Ours (Adaptive)", zorder=3
    )
    for i, method in enumerate(methods):
        axes[1].annotate(
            method.replace("\n", " "),
            (depth_saved_pct[i], mrrs[i]),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9
        )
    axes[1].set_xlabel("Expected Depth Reduction (%)", fontsize=12)
    axes[1].set_ylabel("MRR (%)", fontsize=12)
    axes[1].set_title("Quality vs Efficiency Trade-off", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 4: Exit distribution
# ---------------------------------------------------------------------------

def plot_exit_distribution_enhanced(exit_dist: dict, save_path: str = "exit_distribution.png"):
    """
    Bar chart showing the distribution of dominant exit layers.

    Args:
        exit_dist (dict): Maps exit layer (int) -> percentage (float).
        save_path (str): Output file path. Default: 'exit_distribution.png'.
    """
    layers = list(exit_dist.keys())
    percentages = list(exit_dist.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "mediumseagreen", "coral"]
    bars = ax.bar(layers, percentages, color=colors[: len(layers)], edgecolor="black", linewidth=1.5)

    ax.set_xlabel("Exit Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=14, fontweight="bold")
    ax.set_title("Early-Exit Distribution", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, pct in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: {save_path}")


# ---------------------------------------------------------------------------
# LaTeX results table
# ---------------------------------------------------------------------------

def print_latex_table(your_results: dict):
    """
    Print a LaTeX-formatted results table to stdout.

    Args:
        your_results (dict): Must contain 'mrr', 'hits@20', 'avg_depth'.
    """
    mrr_str = f"{your_results['mrr'] * 100:.1f}"
    h20_str = f"{your_results.get('hits@20', 0.0) * 100:.1f}"
    depth_str = f"{your_results.get('avg_depth', 2.1):.2f}"
    saved_str = f"{(1 - your_results.get('avg_depth', 2.1) / 3.0) * 100:.1f}"

    print("\n" + "=" * 80)
    print("LATEX TABLE (Copy-paste into report)")
    print("=" * 80)
    print(r"""
\begin{table}[h]
\centering
\caption{Performance comparison on Cora dataset (HeaRT evaluation, 500 negatives).}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{MRR (\%)} & \textbf{Hits@20 (\%)} & \textbf{Avg Depth} & \textbf{Exp. Depth Saved} \\
\midrule
Random & 0.2 & 1.0 & - & - \\
Common Neighbors & 10.5 & 30.0 & - & - \\
Resource Allocation & 13.2 & 36.0 & - & - \\
GCN (HeaRT) & 16.6 & 42.0 & 3.0 & 0\% \\
GraphSAGE (HeaRT) & 16.1 & 40.0 & 3.0 & 0\% \\
\midrule"""
          + f"\n\\textbf{{Ours (Adaptive SAGE)}} & \\textbf{{{mrr_str}}} & \\textbf{{{h20_str}}} & \\textbf{{{depth_str}}} & \\textbf{{{saved_str}\\%}} \\\\"
          + r"""
\bottomrule
\end{tabular}
\label{tab:results}
\end{table}
""")
    print("=" * 80)
