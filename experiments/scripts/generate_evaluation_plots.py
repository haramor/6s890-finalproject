"""
Generate Evaluation Plots for Chess Behavioral Cloning Project

This script creates comprehensive visualizations comparing baseline and expert 
models.

Usage:
    python generate_evaluation_plots.py
"""

import sys
import os
from pathlib import Path
import json
from collections import defaultdict

# Install dependencies if needed
required_packages = {
    'tensorboard': 'tensorboard',
    'matplotlib': 'matplotlib',
    'numpy': 'numpy'
}

for package_name, pip_name in required_packages.items():
    try:
        __import__(package_name)
    except ImportError:
        print(f"Installing {pip_name}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Plot styling
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

COLORS = {
    'baseline': '#1f77b4',  # Blue
    'expert': '#ff7f0e'     # Orange
}


def extract_metrics_from_logs(log_dir):
    """Extract all metrics from TensorBoard logs."""
    log_dir = Path(log_dir)

    if not log_dir.exists():
        print(f"Warning: Log directory not found: {log_dir}")
        return {}

    event_files = sorted(log_dir.glob("events.out.tfevents.*"))

    if not event_files:
        print(f"Warning: No event files found in {log_dir}")
        return {}

    print(f"  Found {len(event_files)} event file(s)")

    all_metrics = defaultdict(lambda: {"steps": [], "values": []})

    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(
                str(event_file),
                size_guidance={event_accumulator.SCALARS: 0}
            )
            ea.Reload()

            tags = ea.Tags()['scalars']

            for tag in tags:
                events = ea.Scalars(tag)
                for event in events:
                    all_metrics[tag]["steps"].append(event.step)
                    all_metrics[tag]["values"].append(event.value)

        except Exception as e:
            print(f"    Error processing {event_file.name}: {e}")
            continue

    # Sort by step
    for tag in all_metrics:
        steps = all_metrics[tag]["steps"]
        values = all_metrics[tag]["values"]
        sorted_pairs = sorted(zip(steps, values))
        all_metrics[tag]["steps"] = [s for s, v in sorted_pairs]
        all_metrics[tag]["values"] = [v for s, v in sorted_pairs]

    return dict(all_metrics)


def load_evaluation_results(eval_file):
    """Load evaluation results from JSON."""
    try:
        with open(eval_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {eval_file}: {e}")
        return None


def plot_training_progress(baseline_metrics, expert_metrics, output_dir):
    """Create training progress comparison plot with 4 subplots."""
    print("\nCreating training progress comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress Comparison: Baseline vs Expert', fontsize=16,
fontweight='bold')

    metrics_to_plot = [
        ('train/loss', 'Training Loss', axes[0, 0]),
        ('val/loss', 'Validation Loss', axes[0, 1]),
        ('train/top1_acc', 'Top-1 Accuracy (%)', axes[1, 0]),
        ('train/lr', 'Learning Rate', axes[1, 1])
    ]

    for metric_name, title, ax in metrics_to_plot:
        # Plot baseline
        if metric_name in baseline_metrics:
            data = baseline_metrics[metric_name]
            ax.plot(data['steps'], data['values'],
                    color=COLORS['baseline'], linewidth=2, label='Baseline',
alpha=0.8)

        # Plot expert
        if metric_name in expert_metrics:
            data = expert_metrics[metric_name]
            ax.plot(data['steps'], data['values'],
                    color=COLORS['expert'], linewidth=2, label='Expert', alpha=0.8)

        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()

    output_path = output_dir / 'training_progress_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    return output_path


def plot_sample_complexity(baseline_metrics, expert_metrics, output_dir):
    """Create sample complexity curve."""
    print("\nCreating sample complexity curve...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot baseline
    if 'train/top1_acc' in baseline_metrics:
        data = baseline_metrics['train/top1_acc']
        ax.plot(data['steps'], data['values'],
                color=COLORS['baseline'], linewidth=2, marker='o',
                markersize=4, label='Baseline', alpha=0.8)

    # Plot expert
    if 'train/top1_acc' in expert_metrics:
        data = expert_metrics['train/top1_acc']
        ax.plot(data['steps'], data['values'],
                color=COLORS['expert'], linewidth=2, marker='s',
                markersize=4, label='Expert', alpha=0.8)

    # Target line
    ax.axhline(y=70, color='red', linestyle='--', linewidth=2,
            alpha=0.5, label='Target (70%)')

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Sample Complexity: Accuracy vs Training Steps',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    output_path = output_dir / 'sample_complexity_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    return output_path


def plot_accuracy_comparison(baseline_eval, expert_eval, output_dir):
    """Create bar chart comparing top-k accuracies."""
    print("\nCreating accuracy comparison bar chart...")

    if not baseline_eval or not expert_eval:
        print("  Warning: Missing evaluation results, skipping...")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['top_1', 'top_3', 'top_5']
    labels = ['Top-1', 'Top-3', 'Top-5']
    x = np.arange(len(labels))
    width = 0.35

    baseline_values = [baseline_eval.get(m, 0) for m in metrics]
    expert_values = [expert_eval.get(m, 0) for m in metrics]

    bars1 = ax.bar(x - width/2, baseline_values, width,
                    label='Baseline', color=COLORS['baseline'], alpha=0.8)
    bars2 = ax.bar(x + width/2, expert_values, width,
                    label='Expert', color=COLORS['expert'], alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Accuracy Metric', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = output_dir / 'accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    return output_path


def plot_final_summary(baseline_eval, expert_eval, baseline_metrics, expert_metrics, 
output_dir):
    """Create combined summary figure with table and visualizations."""
    print("\nCreating final performance summary...")

    if not baseline_eval or not expert_eval:
        print("  Warning: Missing evaluation results, skipping...")
        return None

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Top: Summary table
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')

    table_data = [
        ['Metric', 'Baseline', 'Expert', 'Improvement'],
        ['Top-1 Accuracy (%)',
        f"{baseline_eval.get('top_1', 0):.2f}",
        f"{expert_eval.get('top_1', 0):.2f}",
        f"{expert_eval.get('top_1', 0) - baseline_eval.get('top_1', 0):.2f}"],
        ['Top-3 Accuracy (%)',
        f"{baseline_eval.get('top_3', 0):.2f}",
        f"{expert_eval.get('top_3', 0):.2f}",
        f"{expert_eval.get('top_3', 0) - baseline_eval.get('top_3', 0):.2f}"],
        ['Top-5 Accuracy (%)',
        f"{baseline_eval.get('top_5', 0):.2f}",
        f"{expert_eval.get('top_5', 0):.2f}",
        f"{expert_eval.get('top_5', 0) - baseline_eval.get('top_5', 0):.2f}"],
        ['Total Samples',
        f"{baseline_eval.get('total_samples', 0):,}",
        f"{expert_eval.get('total_samples', 0):,}",
        '-']
    ]

    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Header styling
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternating row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')

    ax_table.set_title('Performance Summary', fontsize=16, fontweight='bold', pad=20)

    # Bottom left: Training loss comparison
    ax1 = fig.add_subplot(gs[1, 0])
    if 'train/loss' in baseline_metrics:
        data = baseline_metrics['train/loss']
        ax1.plot(data['steps'], data['values'],
                color=COLORS['baseline'], linewidth=2, label='Baseline', alpha=0.8)
    if 'train/loss' in expert_metrics:
        data = expert_metrics['train/loss']
        ax1.plot(data['steps'], data['values'],
                color=COLORS['expert'], linewidth=2, label='Expert', alpha=0.8)
    ax1.set_xlabel('Steps', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Bottom right: Accuracy comparison bars
    ax2 = fig.add_subplot(gs[1, 1])
    metrics = ['top_1', 'top_3', 'top_5']
    labels = ['Top-1', 'Top-3', 'Top-5']
    x = np.arange(len(labels))
    width = 0.35

    baseline_values = [baseline_eval.get(m, 0) for m in metrics]
    expert_values = [expert_eval.get(m, 0) for m in metrics]

    ax2.bar(x - width/2, baseline_values, width,
            label='Baseline', color=COLORS['baseline'], alpha=0.8)
    ax2.bar(x + width/2, expert_values, width,
            label='Expert', color=COLORS['expert'], alpha=0.8)

    ax2.set_xlabel('Metric', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Final Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Behavioral Cloning Evaluation Summary',
                fontsize=18, fontweight='bold', y=0.98)

    output_path = output_dir / 'final_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    return output_path


def main():
    print("\n" + "="*80)
    print("GENERATING EVALUATION PLOTS")
    print("="*80)

    # Define paths
    base_dir = Path("/workspace/6s890-finalproject")

    baseline_logs = base_dir / "experiments/results/baseline_mixed_skill/logs"
    expert_logs = base_dir / "experiments/results/expert_LE22ct/logs"

    baseline_eval = base_dir / "results/evaluation_safe/baseline_mixed_skill_evaluation.json"
    expert_eval = base_dir / "results/evaluation_safe/expert_LE22ct_evaluation.json"

    output_dir = base_dir / "results/analysis/plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Extract metrics from TensorBoard logs
    print("\nExtracting TensorBoard metrics...")
    print("Baseline:")
    baseline_metrics = extract_metrics_from_logs(baseline_logs)
    print("Expert:")
    expert_metrics = extract_metrics_from_logs(expert_logs)

    # Load evaluation results
    print("\nLoading evaluation results...")
    baseline_eval_data = load_evaluation_results(baseline_eval)
    expert_eval_data = load_evaluation_results(expert_eval)

    # Generate plots
    print("\n" + "-"*80)
    print("GENERATING PLOTS")
    print("-"*80)

    saved_plots = []

    # 1. Training progress comparison
    try:
        path = plot_training_progress(baseline_metrics, expert_metrics, output_dir)
        if path:
            saved_plots.append(path)
    except Exception as e:
        print(f"  Error creating training progress plot: {e}")

    # 2. Sample complexity curve
    try:
        path = plot_sample_complexity(baseline_metrics, expert_metrics, output_dir)
        if path:
            saved_plots.append(path)
    except Exception as e:
        print(f"  Error creating sample complexity plot: {e}")

    # 3. Accuracy comparison
    try:
        path = plot_accuracy_comparison(baseline_eval_data, expert_eval_data, output_dir)
        if path:
            saved_plots.append(path)
    except Exception as e:
        print(f"  Error creating accuracy comparison plot: {e}")

    # 4. Final summary
    try:
        path = plot_final_summary(baseline_eval_data, expert_eval_data, baseline_metrics, expert_metrics, output_dir)
        if path:
            saved_plots.append(path)
    except Exception as e:
        print(f"  Error creating final summary plot: {e}")

    # Summary
    print("\n" + "="*80)
    print("PLOT GENERATION COMPLETE")
    print("="*80)
    print(f"\nGenerated {len(saved_plots)} plots:")
    for plot_path in saved_plots:
        print(f"  ✓ {plot_path}")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
