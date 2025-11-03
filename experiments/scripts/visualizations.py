"""
Visualization Utilities for Chess Behavioral Cloning Evaluation

This module provides functions for creating publication-quality plots:
1. Learning curves (accuracy vs. steps)
2. Sample complexity comparison
3. Stockfish alignment over time
4. Per-position difficulty analysis

Usage:
    from visualizations import plot_learning_curves, plot_stockfish_alignment
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100


def plot_learning_curves(
    results: Dict[str, Dict],
    output_path: str,
    metrics: List[str] = ['top_1', 'top_3', 'loss'],
    title: str = "Learning Curves Comparison"
):
    """
    Plot learning curves for multiple conditions.

    Args:
        results: Dictionary mapping condition names to metric dictionaries
        output_path: Where to save the plot
        metrics: List of metrics to plot
        title: Plot title
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    colors = {
        'baseline_mixed_skill': '#1f77b4',
        'expert_only_2500': '#ff7f0e',
        'game_theoretic_reg': '#2ca02c'
    }

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for condition_name, condition_data in results.items():
            if metric in condition_data:
                steps = condition_data.get('steps', range(len(condition_data[metric])))
                values = condition_data[metric]

                color = colors.get(condition_name, 'gray')
                label = condition_name.replace('_', ' ').title()

                ax.plot(steps, values, label=label, color=color, linewidth=2)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs. Steps')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Learning curves saved to: {output_path}")
    plt.close()


def plot_accuracy_comparison_bars(
    results: Dict[str, Dict],
    output_path: str,
    metrics: List[str] = ['top_1', 'top_3', 'top_5']
):
    """
    Create bar chart comparing final accuracies across conditions.

    Args:
        results: Dictionary mapping condition names to results
        output_path: Where to save the plot
        metrics: Metrics to include in comparison
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.25

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, condition in enumerate(conditions):
        values = [results[condition].get(m, 0) for m in metrics]
        offset = (i - len(conditions)/2 + 0.5) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=condition.replace('_', ' ').title(),
            color=colors[i % len(colors)]
        )

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Comparison Across Conditions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '-').upper() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy comparison saved to: {output_path}")
    plt.close()


def plot_stockfish_alignment(
    results: Dict[str, Dict],
    output_path: str
):
    """
    Plot KL-divergence from Stockfish for each condition.

    Args:
        results: Dictionary with mean_kl and std_kl for each condition
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    conditions = []
    means = []
    stds = []

    for condition, data in results.items():
        if data.get('mean_kl') is not None:
            conditions.append(condition.replace('_', ' ').title())
            means.append(data['mean_kl'])
            stds.append(data.get('std_kl', 0))

    if not conditions:
        print("No Stockfish alignment data to plot")
        return

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    x = np.arange(len(conditions))

    ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(conditions)], alpha=0.7)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Mean KL-Divergence', fontsize=12)
    ax.set_title('Stockfish Alignment (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Stockfish alignment plot saved to: {output_path}")
    plt.close()


def plot_centipawn_loss(
    results: Dict[str, Dict],
    output_path: str
):
    """
    Plot centipawn loss for each condition.

    Args:
        results: Dictionary with mean_cp_loss for each condition
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    conditions = []
    means = []
    stds = []

    for condition, data in results.items():
        if data.get('mean_cp_loss') is not None:
            conditions.append(condition.replace('_', ' ').title())
            means.append(data['mean_cp_loss'])
            stds.append(data.get('std_cp_loss', 0))

    if not conditions:
        print("No centipawn loss data to plot")
        return

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    x = np.arange(len(conditions))

    ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(conditions)], alpha=0.7)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Mean Centipawn Loss', fontsize=12)
    ax.set_title('Average Position Evaluation Loss (Lower is Better)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Centipawn loss plot saved to: {output_path}")
    plt.close()


def create_evaluation_summary_figure(
    results: Dict[str, Dict],
    output_path: str
):
    """
    Create a comprehensive figure with multiple subplots showing all metrics.

    Args:
        results: Dictionary mapping condition names to evaluation results
        output_path: Where to save the figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Top-K Accuracy Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    plot_top_k_comparison(results, ax1)

    # 2. Stockfish Alignment
    ax2 = fig.add_subplot(gs[0, 1])
    plot_kl_comparison(results, ax2)

    # 3. Centipawn Loss
    ax3 = fig.add_subplot(gs[0, 2])
    plot_cp_comparison(results, ax3)

    # 4. Sample Complexity (placeholder)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.text(0.5, 0.5, 'Sample Complexity Curves\n(Requires training data)',
             ha='center', va='center', fontsize=14, color='gray')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.suptitle('Chess Behavioral Cloning: Evaluation Summary',
                 fontsize=18, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive evaluation figure saved to: {output_path}")
    plt.close()


def plot_top_k_comparison(results: Dict[str, Dict], ax):
    """Helper to plot top-k comparison on given axis."""
    conditions = list(results.keys())
    metrics = ['top_1', 'top_3', 'top_5']

    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, condition in enumerate(conditions):
        values = [results[condition].get(m, 0) for m in metrics]
        offset = (i - len(conditions)/2 + 0.5) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=condition.replace('_', '\n').title(),
            color=colors[i % len(colors)],
            alpha=0.8
        )

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Top-K Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '-').upper() for m in metrics])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')


def plot_kl_comparison(results: Dict[str, Dict], ax):
    """Helper to plot KL-divergence comparison on given axis."""
    conditions = []
    means = []
    stds = []

    for condition, data in results.items():
        if data.get('mean_kl') is not None:
            conditions.append(condition.replace('_', '\n').title())
            means.append(data['mean_kl'])
            stds.append(data.get('std_kl', 0))

    if not conditions:
        ax.text(0.5, 0.5, 'No Stockfish\nAlignment Data',
                ha='center', va='center', fontsize=12, color='gray')
        ax.axis('off')
        return

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    x = np.arange(len(conditions))

    ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(conditions)], alpha=0.8)
    ax.set_ylabel('KL-Divergence')
    ax.set_title('Stockfish Alignment\n(Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')


def plot_cp_comparison(results: Dict[str, Dict], ax):
    """Helper to plot centipawn loss comparison on given axis."""
    conditions = []
    means = []
    stds = []

    for condition, data in results.items():
        if data.get('mean_cp_loss') is not None:
            conditions.append(condition.replace('_', '\n').title())
            means.append(data['mean_cp_loss'])
            stds.append(data.get('std_cp_loss', 0))

    if not conditions:
        ax.text(0.5, 0.5, 'No Centipawn\nLoss Data',
                ha='center', va='center', fontsize=12, color='gray')
        ax.axis('off')
        return

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    x = np.arange(len(conditions))

    ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(conditions)], alpha=0.8)
    ax.set_ylabel('Centipawns')
    ax.set_title('Position Evaluation Loss\n(Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')


def load_and_visualize_results(results_dir: str, output_dir: str):
    """
    Load all evaluation results and create visualizations.

    Args:
        results_dir: Directory containing evaluation JSON files
        output_dir: Directory to save visualizations
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("LOADING AND VISUALIZING RESULTS")
    print("="*70)

    # Load all JSON results
    all_results = {}
    for json_file in results_dir.glob("*_evaluation.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            condition_name = data.get('config_name', json_file.stem)
            all_results[condition_name] = data
            print(f"Loaded: {json_file.name}")

    if not all_results:
        print("\nNo evaluation results found!")
        print(f"Looking in: {results_dir}")
        return

    print(f"\nFound {len(all_results)} condition(s)")

    # Create visualizations
    print("\nGenerating visualizations...")

    # Accuracy comparison
    plot_accuracy_comparison_bars(
        all_results,
        str(output_dir / 'accuracy_comparison.png')
    )

    # Stockfish alignment
    plot_stockfish_alignment(
        all_results,
        str(output_dir / 'stockfish_alignment.png')
    )

    # Centipawn loss
    plot_centipawn_loss(
        all_results,
        str(output_dir / 'centipawn_loss.png')
    )

    # Comprehensive summary
    create_evaluation_summary_figure(
        all_results,
        str(output_dir / 'evaluation_summary.png')
    )

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualizations.py <results_dir> [output_dir]")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/visualizations"

    load_and_visualize_results(results_dir, output_dir)
