"""
Generate Evaluation Plots - Fixed for Your Setup
Works with results in: /workspace/6s890-finalproject/experiments/scripts/eval_results/
"""

import sys
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
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "--break-system-packages"])

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Plot styling
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

COLORS = {
    'baseline': '#2E86AB',
    'expert': '#A23B72',
    'game_theoretic_qre': '#F18F01'
}


def extract_metrics_from_logs(log_dir):
    """Extract metrics from TensorBoard logs."""
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

    # Sort by step and remove duplicates
    for tag in all_metrics:
        steps = all_metrics[tag]["steps"]
        values = all_metrics[tag]["values"]
        
        step_value_dict = {}
        for s, v in zip(steps, values):
            step_value_dict[s] = v
        
        sorted_steps = sorted(step_value_dict.keys())
        all_metrics[tag]["steps"] = sorted_steps
        all_metrics[tag]["values"] = [step_value_dict[s] for s in sorted_steps]

    return dict(all_metrics)


def load_evaluation_results(eval_file):
    """Load evaluation results from JSON."""
    try:
        with open(eval_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {eval_file}: {e}")
        return None


def plot_training_progress(metrics_dict, output_dir):
    """Create training progress comparison plot."""
    print("\nCreating training progress comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress Comparison', fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ('train/loss', 'Training Loss', axes[0, 0]),
        ('train/top1_acc', 'Top-1 Accuracy (%)', axes[0, 1]),
        ('train/top3_acc', 'Top-3 Accuracy (%)', axes[1, 0]),
        ('train/lr', 'Learning Rate', axes[1, 1])
    ]

    for metric_name, title, ax in metrics_to_plot:
        plotted = False
        
        for exp_name, metrics in metrics_dict.items():
            if metric_name in metrics:
                data = metrics[metric_name]
                color = COLORS.get(exp_name, '#333333')
                label = exp_name.replace('_', ' ').title()
                ax.plot(data['steps'], data['values'],
                        color=color, linewidth=2, label=label, alpha=0.8)
                plotted = True

        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        if plotted:
            ax.legend()

    plt.tight_layout()

    output_path = output_dir / 'training_progress_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    return output_path


def plot_sample_complexity(metrics_dict, output_dir):
    """Create sample complexity curve."""
    print("\nCreating sample complexity curve...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Find a reasonable target based on actual data
    all_max_values = []
    for exp_name, metrics in metrics_dict.items():
        if 'train/top1_acc' in metrics:
            all_max_values.append(max(metrics['train/top1_acc']['values']))
    
    if not all_max_values:
        print("  Warning: No top1_acc data found, skipping...")
        return None
    
    # Use 80% of the minimum max as target
    target_accuracy = min(all_max_values) * 0.80
    print(f"  Target accuracy: {target_accuracy:.1f}%")

    steps_to_target = {}
    
    for exp_name, metrics in metrics_dict.items():
        if 'train/top1_acc' in metrics:
            data = metrics['train/top1_acc']
            color = COLORS.get(exp_name, '#333333')
            label = exp_name.replace('_', ' ').title()
            
            ax.plot(data['steps'], data['values'],
                    color=color, linewidth=2.5, marker='o',
                    markersize=3, label=label, alpha=0.8)
            
            # Find step where target is reached
            for step, val in zip(data['steps'], data['values']):
                if val >= target_accuracy:
                    steps_to_target[exp_name] = step
                    ax.axvline(x=step, color=color, linestyle=':', 
                              linewidth=2, alpha=0.6)
                    break

    # Target line
    ax.axhline(y=target_accuracy, color='#E63946', linestyle='--', linewidth=2.5,
            alpha=0.7, label=f'Target ({target_accuracy:.1f}%)')

    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Sample Complexity: Steps to Reach {target_accuracy:.1f}% Accuracy',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)

    # Add annotation box
    if steps_to_target:
        text_lines = ["Steps to Target:"]
        for model, steps in steps_to_target.items():
            text_lines.append(f"{model.replace('_', ' ').title()}: {steps:,}")
        
        textstr = '\n'.join(text_lines)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    output_path = output_dir / 'sample_complexity_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    return output_path, steps_to_target


def plot_accuracy_comparison(eval_results_dict, output_dir):
    """Create bar chart comparing accuracies."""
    print("\nCreating accuracy comparison bar chart...")

    if not eval_results_dict or len(eval_results_dict) == 0:
        print("  Warning: No evaluation results, skipping...")
        return None

    fig, ax = plt.subplots(figsize=(12, 7))

    metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
    labels = ['Top-1', 'Top-3', 'Top-5']
    x = np.arange(len(labels))
    width = 0.25

    exp_names = list(eval_results_dict.keys())
    
    for i, exp_name in enumerate(exp_names):
        results = eval_results_dict[exp_name]
        values = [results.get(m, 0) for m in metrics]
        color = COLORS.get(exp_name, '#333333')
        label = exp_name.replace('_', ' ').title()
        
        offset = (i - len(exp_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Accuracy Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison (vs Ground Truth)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 100)

    plt.tight_layout()

    output_path = output_dir / 'accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    return output_path


def plot_stockfish_alignment(eval_results_dict, output_dir, depths=[5, 10, 15]):
    """Create Stockfish alignment comparison."""
    print("\nCreating Stockfish alignment comparison...")

    if not eval_results_dict or len(eval_results_dict) == 0:
        print("  Warning: No evaluation results, skipping...")
        return None

    # Check which depths are available
    available_depths = []
    for depth in depths:
        for results in eval_results_dict.values():
            if f'sf_agreement_depth{depth}' in results:
                available_depths.append(depth)
                break
    
    if not available_depths:
        print("  Warning: No Stockfish alignment data found, skipping...")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Stockfish Alignment Analysis', fontsize=16, fontweight='bold')

    exp_names = list(eval_results_dict.keys())
    
    # Plot 1: Exact Agreement
    ax = axes[0]
    x = np.arange(len(available_depths))
    width = 0.25
    
    for i, exp_name in enumerate(exp_names):
        results = eval_results_dict[exp_name]
        agreement = [results.get(f'sf_agreement_depth{d}', 0) for d in available_depths]
        color = COLORS.get(exp_name, '#333333')
        label = exp_name.replace('_', ' ').title()
        
        offset = (i - len(exp_names)/2 + 0.5) * width
        ax.bar(x + offset, agreement, width, label=label, color=color, alpha=0.8)
    
    ax.set_xlabel('Stockfish Depth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Agreement (%)', fontsize=11, fontweight='bold')
    ax.set_title('Exact Agreement', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Depth {d}' for d in available_depths])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 100)
    
    # Plot 2: Model in SF Top-5
    ax = axes[1]
    for i, exp_name in enumerate(exp_names):
        results = eval_results_dict[exp_name]
        in_sf = [results.get(f'model_in_sf_top5_depth{d}', 0) for d in available_depths]
        color = COLORS.get(exp_name, '#333333')
        label = exp_name.replace('_', ' ').title()
        
        offset = (i - len(exp_names)/2 + 0.5) * width
        ax.bar(x + offset, in_sf, width, label=label, color=color, alpha=0.8)
    
    ax.set_xlabel('Stockfish Depth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Model in SF Top-5', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Depth {d}' for d in available_depths])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 100)
    
    # Plot 3: SF in Model Top-5
    ax = axes[2]
    for i, exp_name in enumerate(exp_names):
        results = eval_results_dict[exp_name]
        sf_in_model = [results.get(f'sf_in_model_top5_depth{d}', 0) for d in available_depths]
        color = COLORS.get(exp_name, '#333333')
        label = exp_name.replace('_', ' ').title()
        
        offset = (i - len(exp_names)/2 + 0.5) * width
        ax.bar(x + offset, sf_in_model, width, label=label, color=color, alpha=0.8)
    
    ax.set_xlabel('Stockfish Depth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('SF in Model Top-5', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Depth {d}' for d in available_depths])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 100)

    plt.tight_layout()

    output_path = output_dir / 'stockfish_alignment_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    return output_path


def main():
    print("\n" + "="*80)
    print("GENERATING EVALUATION PLOTS")
    print("="*80)

    # Paths - update these to match your actual structure
    base_dir = Path("/workspace/6s890-finalproject")
    
    # Check both possible locations for results
    eval_dir_1 = base_dir / "experiments/scripts/eval_results"
    eval_dir_2 = base_dir / "results/evaluation"
    
    if eval_dir_1.exists():
        eval_dir = eval_dir_1
    elif eval_dir_2.exists():
        eval_dir = eval_dir_2
    else:
        print(f"ERROR: Could not find evaluation results in:")
        print(f"  {eval_dir_1}")
        print(f"  {eval_dir_2}")
        return
    
    output_dir = eval_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluation results: {eval_dir}")
    print(f"Output directory: {output_dir}")

    # Load logs and results
    experiments = {
        'baseline': base_dir / "experiments/results/baseline_mixed_skill/logs",
        'expert': base_dir / "experiments/results/expert_LE22ct/logs",
        'game_theoretic_qre': base_dir / "experiments/results/game_theoretic_qre/logs"
    }

    # Extract training metrics
    print("\nExtracting training metrics from logs...")
    all_metrics = {}
    for exp_name, log_dir in experiments.items():
        if log_dir.exists():
            print(f"{exp_name}:")
            all_metrics[exp_name] = extract_metrics_from_logs(log_dir)
        else:
            print(f"{exp_name}: Not found (skipping)")

    # Load evaluation results
    print("\nLoading evaluation results...")
    eval_results = {}
    for exp_name in experiments.keys():
        eval_file = eval_dir / f'{exp_name}_eval.json'
        if eval_file.exists():
            result = load_evaluation_results(eval_file)
            if result:
                eval_results[exp_name] = result
                print(f"  ✓ Loaded: {exp_name}")

    # Generate plots
    print("\n" + "-"*80)
    print("GENERATING PLOTS")
    print("-"*80)

    saved_plots = []

    # 1. Training progress
    if all_metrics:
        try:
            path = plot_training_progress(all_metrics, output_dir)
            if path:
                saved_plots.append(path)
        except Exception as e:
            print(f"  Error creating training progress plot: {e}")

    # 2. Sample complexity
    if all_metrics:
        try:
            result = plot_sample_complexity(all_metrics, output_dir)
            if result:
                path, _ = result
                saved_plots.append(path)
        except Exception as e:
            print(f"  Error creating sample complexity plot: {e}")

    # 3. Accuracy comparison
    if eval_results:
        try:
            path = plot_accuracy_comparison(eval_results, output_dir)
            if path:
                saved_plots.append(path)
        except Exception as e:
            print(f"  Error creating accuracy comparison plot: {e}")

    # 4. Stockfish alignment
    if eval_results:
        try:
            path = plot_stockfish_alignment(eval_results, output_dir)
            if path:
                saved_plots.append(path)
        except Exception as e:
            print(f"  Error creating Stockfish alignment plot: {e}")

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
