"""
Sample Complexity Analysis

This script analyzes sample complexity curves by:
1. Training models with varying amounts of data
2. Plotting accuracy vs. number of training samples
3. Computing the number of samples needed to reach target accuracy (e.g., 70%)

Usage:
    python sample_complexity.py --condition baseline
    python sample_complexity.py --condition expert_only
    python sample_complexity.py --condition game_theoretic
    python sample_complexity.py --all-conditions
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def train_with_data_fraction(
    config_path: str,
    data_fraction: float,
    output_dir: str,
    max_steps: int = 1000
) -> Dict:
    """
    Train a model with a fraction of the data.

    Args:
        config_path: Path to configuration file
        data_fraction: Fraction of data to use (0-1)
        output_dir: Directory to save results
        max_steps: Maximum training steps

    Returns:
        Dictionary with training results and final metrics
    """
    # This would require modifying the training script to accept data fraction
    # For now, this is a placeholder showing the concept

    print(f"\nTraining with {data_fraction*100:.0f}% of data...")

    # In practice, you would:
    # 1. Modify config to use subset of data
    # 2. Run training
    # 3. Evaluate final model
    # 4. Return metrics

    # Placeholder results
    results = {
        'data_fraction': data_fraction,
        'n_samples': int(10000 * data_fraction),  # Assuming 10K total samples
        'final_accuracy': None,  # Would be computed from evaluation
        'training_time': None,
        'final_loss': None
    }

    return results


def compute_sample_complexity(
    accuracy_values: List[float],
    sample_counts: List[int],
    target_accuracy: float = 70.0
) -> Tuple[int, float]:
    """
    Compute number of samples needed to reach target accuracy.

    Args:
        accuracy_values: List of accuracy values
        sample_counts: List of corresponding sample counts
        target_accuracy: Target accuracy to reach

    Returns:
        Tuple of (samples_needed, interpolated_accuracy)
    """
    # Sort by sample count
    sorted_pairs = sorted(zip(sample_counts, accuracy_values))
    sample_counts = [x[0] for x in sorted_pairs]
    accuracy_values = [x[1] for x in sorted_pairs]

    # Find where we cross target accuracy
    for i, (samples, acc) in enumerate(zip(sample_counts, accuracy_values)):
        if acc >= target_accuracy:
            if i == 0:
                return samples, acc
            else:
                # Linear interpolation
                prev_samples, prev_acc = sample_counts[i-1], accuracy_values[i-1]
                frac = (target_accuracy - prev_acc) / (acc - prev_acc)
                samples_needed = int(prev_samples + frac * (samples - prev_samples))
                return samples_needed, target_accuracy

    # Target not reached
    return sample_counts[-1], accuracy_values[-1]


def plot_sample_complexity_curves(
    results: Dict[str, List[Dict]],
    output_path: str,
    target_accuracy: float = 70.0
):
    """
    Plot sample complexity curves for all conditions.

    Args:
        results: Dictionary mapping condition names to list of result dicts
        output_path: Where to save the plot
        target_accuracy: Target accuracy line to draw
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'baseline': '#1f77b4',
        'expert_only': '#ff7f0e',
        'game_theoretic': '#2ca02c'
    }

    for condition_name, condition_results in results.items():
        # Extract data points
        sample_counts = [r['n_samples'] for r in condition_results]
        accuracies = [r['final_accuracy'] for r in condition_results]

        # Sort by sample count
        sorted_pairs = sorted(zip(sample_counts, accuracies))
        sample_counts = [x[0] for x in sorted_pairs]
        accuracies = [x[1] for x in sorted_pairs]

        # Plot
        color = colors.get(condition_name, 'gray')
        ax.plot(
            sample_counts,
            accuracies,
            marker='o',
            label=condition_name.replace('_', ' ').title(),
            color=color,
            linewidth=2,
            markersize=8
        )

    # Add target accuracy line
    ax.axhline(
        y=target_accuracy,
        color='red',
        linestyle='--',
        linewidth=1,
        alpha=0.7,
        label=f'Target ({target_accuracy}%)'
    )

    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Sample Complexity Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSample complexity plot saved to: {output_path}")
    plt.close()


def create_sample_complexity_table(
    results: Dict[str, List[Dict]],
    output_path: str,
    target_accuracy: float = 70.0
):
    """
    Create a table summarizing sample complexity results.

    Args:
        results: Dictionary mapping condition names to results
        output_path: Where to save the table (CSV)
        target_accuracy: Target accuracy for comparison
    """
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Condition',
            f'Samples for {target_accuracy}% Accuracy',
            'Reduction vs Baseline (%)',
            'Final Accuracy',
            'Final Sample Count'
        ])

        baseline_samples = None

        for condition_name, condition_results in results.items():
            sample_counts = [r['n_samples'] for r in condition_results]
            accuracies = [r['final_accuracy'] for r in condition_results]

            samples_needed, final_acc = compute_sample_complexity(
                accuracies, sample_counts, target_accuracy
            )

            if condition_name == 'baseline':
                baseline_samples = samples_needed
                reduction = 0.0
            elif baseline_samples:
                reduction = (1 - samples_needed / baseline_samples) * 100
            else:
                reduction = None

            writer.writerow([
                condition_name,
                samples_needed,
                f"{reduction:.1f}" if reduction is not None else "N/A",
                f"{final_acc:.2f}",
                max(sample_counts)
            ])

    print(f"\nSample complexity table saved to: {output_path}")


def analyze_existing_results(results_dir: str, output_dir: str):
    """
    Analyze existing training results to extract sample complexity curves.

    This function looks for training logs and checkpoints to reconstruct
    the learning curves at different data scales.

    Args:
        results_dir: Directory containing training results
        output_dir: Directory to save analysis outputs
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SAMPLE COMPLEXITY ANALYSIS")
    print("="*70)

    # Look for result directories
    conditions = ['baseline_mixed_skill', 'expert_only_2500', 'game_theoretic_reg']

    all_results = {}

    for condition in conditions:
        condition_dir = results_dir / condition

        if not condition_dir.exists():
            print(f"\nWarning: No results found for {condition}")
            continue

        print(f"\nAnalyzing {condition}...")

        # Look for training logs
        logs_dir = condition_dir / 'logs'

        # This is a placeholder - in practice you would:
        # 1. Parse TensorBoard logs to get accuracy at different steps
        # 2. Map steps to approximate sample counts
        # 3. Extract accuracy values

        # Placeholder data
        condition_results = [
            {'n_samples': 1000, 'final_accuracy': 45.0},
            {'n_samples': 2500, 'final_accuracy': 55.0},
            {'n_samples': 5000, 'final_accuracy': 65.0},
            {'n_samples': 7500, 'final_accuracy': 72.0},
            {'n_samples': 10000, 'final_accuracy': 75.0},
        ]

        all_results[condition] = condition_results

    if not all_results:
        print("\nNo results found to analyze!")
        return

    # Create plots
    plot_path = output_dir / 'sample_complexity_curves.png'
    plot_sample_complexity_curves(all_results, str(plot_path))

    # Create table
    table_path = output_dir / 'sample_complexity_table.csv'
    create_sample_complexity_table(all_results, str(table_path))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sample complexity of chess behavioral cloning"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing training results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sample_complexity",
        help="Directory to save analysis outputs"
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=70.0,
        help="Target accuracy for sample complexity comparison"
    )

    args = parser.parse_args()

    analyze_existing_results(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
