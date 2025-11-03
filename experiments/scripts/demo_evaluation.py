"""
Demo Evaluation Script

This script demonstrates the evaluation pipeline using mock data.
Use this to test the evaluation system before running on real trained models.

Usage:
    python demo_evaluation.py
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def create_mock_results():
    """
    Create mock evaluation results for demonstration.

    These are hypothetical results based on the project hypotheses:
    - Baseline: Standard performance
    - Expert-only: 30-50% reduction in sample complexity
    - Game-theoretic: Additional 15-25% reduction
    """

    # Mock results for three conditions
    results = {
        'baseline_mixed_skill': {
            'config_name': 'baseline_mixed_skill',
            'experiment_type': 'baseline',
            'top_1': 68.5,
            'top_3': 85.2,
            'top_5': 91.7,
            'mean_kl': 1.85,
            'std_kl': 0.52,
            'n_positions': 1000,
            'mean_cp_loss': 95.3,
            'std_cp_loss': 48.7
        },
        'expert_only_2500': {
            'config_name': 'expert_only_2500',
            'experiment_type': 'expert_only',
            'top_1': 74.2,  # Better accuracy
            'top_3': 89.1,
            'top_5': 94.3,
            'mean_kl': 1.52,  # Better Stockfish alignment
            'std_kl': 0.41,
            'n_positions': 1000,
            'mean_cp_loss': 78.6,  # Lower centipawn loss
            'std_cp_loss': 42.1
        },
        'game_theoretic_reg': {
            'config_name': 'game_theoretic_reg',
            'experiment_type': 'game_theoretic',
            'top_1': 76.8,  # Best accuracy
            'top_3': 90.5,
            'top_5': 95.1,
            'mean_kl': 1.28,  # Best Stockfish alignment
            'std_kl': 0.35,
            'n_positions': 1000,
            'mean_cp_loss': 71.2,  # Lowest centipawn loss
            'std_cp_loss': 38.9
        }
    }

    return results


def create_mock_sample_complexity_data():
    """Create mock sample complexity curves."""

    # Sample counts
    sample_counts = [1000, 2500, 5000, 7500, 10000, 15000, 20000]

    # Mock learning curves (hypothetical based on project hypotheses)
    baseline_acc = [35.2, 48.5, 60.3, 66.8, 68.5, 70.2, 71.5]
    expert_acc = [42.8, 58.3, 68.7, 73.1, 74.2, 75.8, 76.5]
    gt_acc = [45.6, 62.1, 72.4, 75.9, 76.8, 78.1, 78.9]

    return {
        'sample_counts': sample_counts,
        'baseline': baseline_acc,
        'expert_only': expert_acc,
        'game_theoretic': gt_acc
    }


def plot_mock_accuracy_comparison(results, output_path):
    """Create bar chart of top-k accuracies."""
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = list(results.keys())
    metrics = ['top_1', 'top_3', 'top_5']

    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, condition in enumerate(conditions):
        values = [results[condition].get(m, 0) for m in metrics]
        offset = (i - len(conditions)/2 + 0.5) * width

        label = condition.replace('_', ' ').title()
        ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=colors[i],
            alpha=0.8
        )

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Top-K Accuracy Comparison (Mock Data)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Top-1', 'Top-3', 'Top-5'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_mock_sample_complexity(data, output_path):
    """Create sample complexity curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sample_counts = data['sample_counts']

    # Plot curves
    ax.plot(sample_counts, data['baseline'],
            marker='o', label='Baseline (Mixed Skill)',
            color='#1f77b4', linewidth=2, markersize=8)
    ax.plot(sample_counts, data['expert_only'],
            marker='s', label='Expert-Only (ELO 2500+)',
            color='#ff7f0e', linewidth=2, markersize=8)
    ax.plot(sample_counts, data['game_theoretic'],
            marker='^', label='Game-Theoretic Regularization',
            color='#2ca02c', linewidth=2, markersize=8)

    # Add target line
    ax.axhline(y=70, color='red', linestyle='--', linewidth=1,
               alpha=0.7, label='Target (70%)')

    ax.set_xlabel('Number of Training Games', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Sample Complexity Comparison (Mock Data)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(30, 85)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def compute_sample_complexity_stats(data, target_accuracy=70.0):
    """Compute how many samples needed to reach target."""

    def interpolate_samples(sample_counts, accuracies, target):
        """Linear interpolation to find samples needed."""
        for i, acc in enumerate(accuracies):
            if acc >= target:
                if i == 0:
                    return sample_counts[0], acc
                else:
                    # Linear interpolation
                    prev_samples = sample_counts[i-1]
                    prev_acc = accuracies[i-1]
                    frac = (target - prev_acc) / (acc - prev_acc)
                    samples = int(prev_samples + frac * (sample_counts[i] - prev_samples))
                    return samples, target
        return sample_counts[-1], accuracies[-1]

    sample_counts = data['sample_counts']

    results = {}
    for condition in ['baseline', 'expert_only', 'game_theoretic']:
        samples, acc = interpolate_samples(
            sample_counts,
            data[condition],
            target_accuracy
        )
        results[condition] = {
            'samples_for_target': samples,
            'final_accuracy': acc
        }

    # Compute reductions
    baseline_samples = results['baseline']['samples_for_target']
    expert_reduction = (1 - results['expert_only']['samples_for_target'] / baseline_samples) * 100
    gt_reduction = (1 - results['game_theoretic']['samples_for_target'] / baseline_samples) * 100

    results['expert_only']['reduction_vs_baseline'] = expert_reduction
    results['game_theoretic']['reduction_vs_baseline'] = gt_reduction

    return results


def create_summary_table(evaluation_results, complexity_stats, output_path):
    """Create a markdown summary table."""

    lines = [
        "# Evaluation Results Summary (Mock Data)",
        "",
        "## Top-K Accuracy Comparison",
        "",
        "| Condition | Top-1 | Top-3 | Top-5 |",
        "|-----------|-------|-------|-------|"
    ]

    for condition, data in evaluation_results.items():
        name = condition.replace('_', ' ').title()
        lines.append(
            f"| {name} | {data['top_1']:.1f}% | {data['top_3']:.1f}% | {data['top_5']:.1f}% |"
        )

    lines.extend([
        "",
        "## Stockfish Alignment (KL-Divergence)",
        "",
        "| Condition | Mean KL | Std KL |",
        "|-----------|---------|--------|"
    ])

    for condition, data in evaluation_results.items():
        name = condition.replace('_', ' ').title()
        lines.append(
            f"| {name} | {data['mean_kl']:.2f} | {data['std_kl']:.2f} |"
        )

    lines.extend([
        "",
        "## Sample Complexity (for 70% Accuracy)",
        "",
        "| Condition | Samples Needed | Reduction vs Baseline |",
        "|-----------|----------------|----------------------|"
    ])

    for condition in ['baseline', 'expert_only', 'game_theoretic']:
        name = condition.replace('_', ' ').title()
        stats = complexity_stats[condition]
        samples = stats['samples_for_target']
        reduction = stats.get('reduction_vs_baseline', 0)
        lines.append(
            f"| {name} | {samples:,} | {reduction:.1f}% |"
        )

    lines.extend([
        "",
        "## Key Findings (Mock Data)",
        "",
        f"1. **Expert-only approach** achieves {complexity_stats['expert_only']['reduction_vs_baseline']:.1f}% reduction in sample complexity",
        f"2. **Game-theoretic regularization** achieves {complexity_stats['game_theoretic']['reduction_vs_baseline']:.1f}% total reduction",
        f"3. **Top-1 accuracy** improves from {evaluation_results['baseline_mixed_skill']['top_1']:.1f}% (baseline) to {evaluation_results['game_theoretic_reg']['top_1']:.1f}% (game-theoretic)",
        "",
        "---",
        "*Note: These are mock results for demonstration purposes*"
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  ✓ Saved: {output_path}")


def main():
    print("="*70)
    print("DEMO EVALUATION - Mock Data")
    print("="*70)
    print("\nThis script demonstrates the evaluation pipeline using mock data.")
    print("Replace with real evaluation results from trained models.\n")

    # Create output directory
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)

    # Generate mock data
    print("Generating mock evaluation results...")
    evaluation_results = create_mock_results()
    sample_complexity_data = create_mock_sample_complexity_data()

    # Save JSON results
    print("\nSaving JSON results...")
    for condition, data in evaluation_results.items():
        json_path = output_dir / f"{condition}_evaluation.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Saved: {json_path}")

    # Create visualizations
    print("\nGenerating visualizations...")

    plot_mock_accuracy_comparison(
        evaluation_results,
        output_dir / "accuracy_comparison.png"
    )

    plot_mock_sample_complexity(
        sample_complexity_data,
        output_dir / "sample_complexity_curves.png"
    )

    # Compute sample complexity statistics
    print("\nComputing sample complexity statistics...")
    complexity_stats = compute_sample_complexity_stats(sample_complexity_data)

    # Create summary
    print("\nCreating summary table...")
    create_summary_table(
        evaluation_results,
        complexity_stats,
        output_dir / "evaluation_summary.md"
    )

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY (Mock Data)")
    print("="*70)
    print("\nTop-1 Accuracy:")
    for condition, data in evaluation_results.items():
        name = condition.replace('_', ' ').title()
        print(f"  {name:40s} {data['top_1']:5.1f}%")

    print("\nSample Complexity (for 70% accuracy):")
    for condition in ['baseline', 'expert_only', 'game_theoretic']:
        name = condition.replace('_', ' ').title()
        stats = complexity_stats[condition]
        reduction = stats.get('reduction_vs_baseline', 0)
        print(f"  {name:40s} {stats['samples_for_target']:6,} games ({reduction:+5.1f}%)")

    print("\n" + "="*70)
    print("KEY FINDINGS (Hypothetical)")
    print("="*70)
    print(f"\n1. Expert-only reduces sample complexity by {complexity_stats['expert_only']['reduction_vs_baseline']:.1f}%")
    print(f"2. Game-theoretic achieves {complexity_stats['game_theoretic']['reduction_vs_baseline']:.1f}% total reduction")
    print(f"3. This supports the project hypothesis of 30-50% + 15-25% reductions")

    print("\n" + "="*70)
    print("OUTPUTS")
    print("="*70)
    print(f"\nAll files saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - *.json: Raw evaluation results")
    print("  - accuracy_comparison.png: Bar chart of accuracies")
    print("  - sample_complexity_curves.png: Learning curves")
    print("  - evaluation_summary.md: Summary table")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Run real evaluations:")
    print("   python scripts/evaluate.py --all-conditions")
    print("\n2. Replace mock data with real results")
    print("\n3. Use these figures as templates for your report")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
