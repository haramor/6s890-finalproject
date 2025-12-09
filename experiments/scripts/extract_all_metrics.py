"""
Extract All Metrics from TensorBoard Logs

This script extracts all metrics from TensorBoard event files and outputs them
in both JSON format and printed to stdout for easy copy-pasting.

Usage:
    python extract_all_metrics.py
"""

import sys
import os
from pathlib import Path
import json
from collections import defaultdict

# Try to import tensorboard, install if needed
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("TensorBoard not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
    from tensorboard.backend.event_processing import event_accumulator


def extract_metrics_from_logs(log_dir):
    """
    Extract all metrics from TensorBoard logs in a directory.

    Args:
        log_dir: Path to directory containing TensorBoard event files

    Returns:
        Dictionary mapping metric names to {steps: [...], values: [...]}
    """
    log_dir = Path(log_dir)

    if not log_dir.exists():
        print(f"Warning: Log directory not found: {log_dir}")
        return {}

    # Find all event files
    event_files = sorted(log_dir.glob("events.out.tfevents.*"))

    if not event_files:
        print(f"Warning: No event files found in {log_dir}")
        return {}

    print(f"Found {len(event_files)} event file(s) in {log_dir}")

    # Aggregate metrics from all event files
    all_metrics = defaultdict(lambda: {"steps": [], "values": []})

    for event_file in event_files:
        print(f"  Processing: {event_file.name}")

        try:
            # Load event file
            ea = event_accumulator.EventAccumulator(
                str(event_file),
                size_guidance={
                    event_accumulator.SCALARS: 0,  # 0 means load all
                }
            )
            ea.Reload()

            # Get all scalar tags (metrics)
            tags = ea.Tags()['scalars']

            for tag in tags:
                events = ea.Scalars(tag)
                for event in events:
                    all_metrics[tag]["steps"].append(event.step)
                    all_metrics[tag]["values"].append(event.value)

        except Exception as e:
            print(f"  Error processing {event_file.name}: {e}")
            continue

    # Sort by step for each metric
    for tag in all_metrics:
        steps = all_metrics[tag]["steps"]
        values = all_metrics[tag]["values"]

        # Sort by step
        sorted_pairs = sorted(zip(steps, values))
        all_metrics[tag]["steps"] = [s for s, v in sorted_pairs]
        all_metrics[tag]["values"] = [v for s, v in sorted_pairs]

    return dict(all_metrics)


def print_metrics_summary(experiment_name, metrics):
    """Print a summary of extracted metrics."""
    print(f"\n{'='*80}")
    print(f"SUMMARY: {experiment_name}")
    print(f"{'='*80}")

    if not metrics:
        print("No metrics found.")
        return

    print(f"Total metrics extracted: {len(metrics)}\n")

    for metric_name, data in sorted(metrics.items()):
        num_points = len(data["steps"])
        if num_points > 0:
            step_range = f"{min(data['steps'])} to {max(data['steps'])}"
            value_range = f"{min(data['values']):.4f} to {max(data['values']):.4f}"
            print(f"  {metric_name}:")
            print(f"    Data points: {num_points}")
            print(f"    Step range: {step_range}")
            print(f"    Value range: {value_range}")


def print_metrics_data(experiment_name, metrics):
    """Print all metrics data in copy-pasteable format."""
    print(f"\n{'='*80}")
    print(f"RAW DATA: {experiment_name}")
    print(f"{'='*80}\n")

    if not metrics:
        print("No data to display.")
        return

    for metric_name, data in sorted(metrics.items()):
        print(f"{experiment_name} - {metric_name}")
        print(f"  steps: {data['steps']}")
        print(f"  values: {data['values']}")
        print()


def main():
    # Define paths
    base_dir = Path("/workspace/6s890-finalproject")

    experiments = {
        "baseline_mixed_skill": {
            "logs": base_dir / "experiments/results/baseline_mixed_skill/logs",
            "output": base_dir / "results/analysis/baseline_mixed_skill_metrics.json"
        },
        "expert_LE22ct": {
            "logs": base_dir / "experiments/results/expert_LE22ct/logs",
            "output": base_dir / "results/analysis/expert_LE22ct_metrics.json"
        }
    }

    # Create output directory
    output_dir = base_dir / "results/analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Extract metrics for each experiment
    all_experiment_data = {}

    for experiment_name, paths in experiments.items():
        print(f"\n{'#'*80}")
        print(f"# PROCESSING: {experiment_name}")
        print(f"{'#'*80}\n")

        # Extract metrics
        metrics = extract_metrics_from_logs(paths["logs"])
        all_experiment_data[experiment_name] = metrics

        # Save to JSON
        try:
            with open(paths["output"], 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nSaved to: {paths['output']}")
        except Exception as e:
            print(f"\nError saving JSON: {e}")

        # Print summary
        print_metrics_summary(experiment_name, metrics)

    # Print all raw data for copy-pasting
    print("\n\n")
    print("#" * 80)
    print("# COPY-PASTEABLE RAW DATA")
    print("#" * 80)

    for experiment_name, metrics in all_experiment_data.items():
        print_metrics_data(experiment_name, metrics)

    # Final summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print("\nFiles saved:")
    for experiment_name, paths in experiments.items():
        if all_experiment_data[experiment_name]:
            print(f"  - {paths['output']}")

    print("\nYou can now:")
    print("  1. Copy the raw data printed above to use locally")
    print("  2. Download the JSON files for plotting")
    print("  3. Use the data to generate comparison plots")


if __name__ == "__main__":
    main()
