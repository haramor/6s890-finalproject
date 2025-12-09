"""
Generate Comprehensive Evaluation Plots for Chess Behavioral Cloning

Creates visualizations including:
- Training progress comparison
- Sample complexity curves  
- Accuracy comparisons
- Stockfish alignment analysis
- Final summary

Usage:
    python generate_comprehensive_plots.py
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
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "--break-system-packages"])

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Plot styling
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

COLORS = {
    'baseline': '#2E86AB',  # Blue
    'expert': '#A23B72'     # Purple
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

    # Sort by step and remove duplicates
    for tag in all_metrics:
        steps = all_metrics[tag]["steps"]
        values = all_metrics[tag]["values"]
        
        # Use dict to remove duplicates (keep last value for each step)
        step_value_dict = {}
        for s, v in zip(steps, values):
            step_value_dict[s] = v
        
        sorted_steps = sorted(step_value_dict.keys())
        all_metrics[tag]["steps"] = sorted_steps
        all_metrics[tag]["values"] = [step_value_dict[s] for s in sorted_steps]

    return dict(all_metrics)


def analyze_logs_for_target(metrics_dict):
    """Analyze logs to determine realistic target for sample complexity."""
    if 'train/top5_acc' in metrics_dict:
        values = metrics_dict['train/top5_acc']['values']
        max_val = max(values) if values else 0
        final_val = values[-1] if values else 0
        return max_val, final_val
    return 0, 0


def load_evaluation_results(eval_file):
    """Load evaluation results from JSON."""
    try:
        with open(eval_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {eval_file}: {e}")
        return None


def plot_training_progress(baseline_metrics, expert_metrics, output_dir):
    """Create training progress comparison plot."""
    print("\nCreating training progress comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress: Baseline vs Expert', fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ('train/loss', 'Training Loss', axes[0, 0]),
        ('train/top1_acc', 'Top-1 Accuracy (%)', axes[0, 1]),
        ('train/top3_acc', 'Top-3 Accuracy (%)', axes[1, 0]),
        ('train/top5_acc', 'Top-5 Accuracy (%)', axes[1, 1])
    ]

    for metric_name, title, ax in metrics_to_plot:
        plotted = False
        
        # Plot baseline
        if metric_name in baseline_metrics:
            data = baseline_metrics[metric_name]
            ax.plot(data['steps'], data['values'],
                    color=COLORS['baseline'], linewidth=2, label='Baseline',
                    alpha=0.8)
            plotted = True

        # Plot expert
        if metric_name in expert_metrics:
            data = expert_metrics[metric_name]
            ax.plot(data['steps'], data['values'],
                    color=COLORS['expert'], linewidth=2, label='Expert', alpha=0.8)
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


def plot_sample_complexity(baseline_metrics, expert_metrics, output_dir, target_accuracy=None):
    """Create sample complexity curve showing steps to reach target accuracy."""
    print("\nCreating sample complexity curve...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Determine target if not provided
    if target_accuracy is None:
        baseline_max, _ = analyze_logs_for_target(baseline_metrics)
        expert_max, _ = analyze_logs_for_target(expert_metrics)
        target_accuracy = min(baseline_max, expert_max) * 0.85
    
    print(f"  Target accuracy: {target_accuracy:.1f}%")

    steps_to_target = {}
    
    # Plot baseline
    if 'train/top5_acc' in baseline_metrics:
        data = baseline_metrics['train/top5_acc']
        ax.plot(data['steps'], data['values'],
                color=COLORS['baseline'], linewidth=2.5, marker='o',
                markersize=3, label='Baseline', alpha=0.8)
        
        # Find step where target is reached
        for step, val in zip(data['steps'], data['values']):
            if val >= target_accuracy:
                steps_to_target['baseline'] = step
                ax.axvline(x=step, color=COLORS['baseline'], linestyle=':', 
                          linewidth=2, alpha=0.6)
                break

    # Plot expert
    if 'train/top5_acc' in expert_metrics:
        data = expert_metrics['train/top5_acc']
        ax.plot(data['steps'], data['values'],
                color=COLORS['expert'], linewidth=2.5, marker='s',
                markersize=3, label='Expert', alpha=0.8)
        
        # Find step where target is reached
        for step, val in zip(data['steps'], data['values']):
            if val >= target_accuracy:
                steps_to_target['expert'] = step
                ax.axvline(x=step, color=COLORS['expert'], linestyle=':', 
                          linewidth=2, alpha=0.6)
                break

    # Target line
    ax.axhline(y=target_accuracy, color='#E63946', linestyle='--', linewidth=2.5,
            alpha=0.7, label=f'Target ({target_accuracy:.1f}%)')

    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top-5 Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Sample Complexity: Steps to Reach {target_accuracy:.1f}% Top-5 Accuracy',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)

    # Add annotation box with results
    if steps_to_target:
        text_lines = ["Steps to Target:"]
        for model, steps in steps_to_target.items():
            text_lines.append(f"{model.capitalize()}: {steps:,}")
        
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


def plot_accuracy_comparison(baseline_eval, expert_eval, output_dir):
    """Create bar chart comparing accuracies."""
    print("\nCreating accuracy comparison bar chart...")

    if not baseline_eval or not expert_eval:
        print("  Warning: Missing evaluation results, skipping...")
        return None

    fig, ax = plt.subplots(figsize=(12, 7))

    metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
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


def plot_stockfish_alignment(baseline_eval, expert_eval, output_dir):
    """Create Stockfish alignment comparison across depths."""
    print("\nCreating Stockfish alignment comparison...")

    if not baseline_eval or not expert_eval:
        print("  Warning: Missing evaluation results, skipping...")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Stockfish Alignment Analysis', fontsize=16, fontweight='bold')

    depths = [5, 10, 15]
    
    # Plot 1: Exact Agreement
    ax = axes[0]
    baseline_agreement = [baseline_eval.get(f'sf_agreement_depth{d}', 0) for d in depths]
    expert_agreement = [expert_eval.get(f'sf_agreement_depth{d}', 0) for d in depths]
    
    x = np.arange(len(depths))
    width = 0.35
    
    ax.bar(x - width/2, baseline_agreement, width, label='Baseline', 
           color=COLORS['baseline'], alpha=0.8)
    ax.bar(x + width/2, expert_agreement, width, label='Expert',
           color=COLORS['expert'], alpha=0.8)
    
    ax.set_xlabel('Stockfish Depth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Agreement (%)', fontsize=11, fontweight='bold')
    ax.set_title('Exact Agreement', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Depth {d}' for d in depths])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 100)
    
    # Plot 2: Model in SF Top-5
    ax = axes[1]
    baseline_in_sf = [baseline_eval.get(f'model_in_sf_top5_depth{d}', 0) for d in depths]
    expert_in_sf = [expert_eval.get(f'model_in_sf_top5_depth{d}', 0) for d in depths]
    
    ax.bar(x - width/2, baseline_in_sf, width, label='Baseline',
           color=COLORS['baseline'], alpha=0.8)
    ax.bar(x + width/2, expert_in_sf, width, label='Expert',
           color=COLORS['expert'], alpha=0.8)
    
    ax.set_xlabel('Stockfish Depth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Model in SF Top-5', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Depth {d}' for d in depths])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 100)
    
    # Plot 3: SF in Model Top-5
    ax = axes[2]
    baseline_sf_in_model = [baseline_eval.get(f'sf_in_model_top5_depth{d}', 0) for d in depths]
    expert_sf_in_model = [expert_eval.get(f'sf_in_model_top5_depth{d}', 0) for d in depths]
    
    ax.bar(x - width/2, baseline_sf_in_model, width, label='Baseline',
           color=COLORS['baseline'], alpha=0.8)
    ax.bar(x + width/2, expert_sf_in_model, width, label='Expert',
           color=COLORS['expert'], alpha=0.8)
    
    ax.set_xlabel('Stockfish Depth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('SF in Model Top-5', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Depth {d}' for d in depths])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 100)

    plt.tight_layout()

    output_path = output_dir / 'stockfish_alignment_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    return output_path


def plot_final_summary(baseline_eval, expert_eval, baseline_metrics, expert_metrics, 
                      output_dir, steps_to_target=None):
    """Create comprehensive summary figure."""
    print("\nCreating final summary figure...")

    if not baseline_eval or not expert_eval:
        print("  Warning: Missing evaluation results, skipping...")
        return None

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Top: Summary table
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')

    table_data = [
        ['Metric', 'Baseline', 'Expert', 'Δ (Expert - Baseline)'],
        ['Samples Evaluated',
         f"{baseline_eval.get('n_evaluated', 0):,}",
         f"{expert_eval.get('n_evaluated', 0):,}",
         '-'],
        ['Legal Move Rate (%)',
         f"{baseline_eval.get('legal_move_rate', 0):.2f}",
         f"{expert_eval.get('legal_move_rate', 0):.2f}",
         f"{expert_eval.get('legal_move_rate', 0) - baseline_eval.get('legal_move_rate', 0):+.2f}"],
        ['Top-1 Accuracy (%)',
         f"{baseline_eval.get('top1_accuracy', 0):.2f}",
         f"{expert_eval.get('top1_accuracy', 0):.2f}",
         f"{expert_eval.get('top1_accuracy', 0) - baseline_eval.get('top1_accuracy', 0):+.2f}"],
        ['Top-5 Accuracy (%)',
         f"{baseline_eval.get('top5_accuracy', 0):.2f}",
         f"{expert_eval.get('top5_accuracy', 0):.2f}",
         f"{expert_eval.get('top5_accuracy', 0) - baseline_eval.get('top5_accuracy', 0):+.2f}"],
        ['SF Agreement @ depth 15 (%)',
         f"{baseline_eval.get('sf_agreement_depth15', 0):.2f}",
         f"{expert_eval.get('sf_agreement_depth15', 0):.2f}",
         f"{expert_eval.get('sf_agreement_depth15', 0) - baseline_eval.get('sf_agreement_depth15', 0):+.2f}"],
    ]
    
    # Add sample complexity if available
    if steps_to_target:
        baseline_steps = steps_to_target.get('baseline', '-')
        expert_steps = steps_to_target.get('expert', '-')
        if isinstance(baseline_steps, int) and isinstance(expert_steps, int):
            reduction = ((baseline_steps - expert_steps) / baseline_steps) * 100
            table_data.append([
                'Steps to Target',
                f"{baseline_steps:,}",
                f"{expert_steps:,}",
                f"{reduction:+.1f}%"
            ])

    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.35, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

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

    # Middle row: Training curves
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
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 1])
    if 'train/top5_acc' in baseline_metrics:
        data = baseline_metrics['train/top5_acc']
        ax2.plot(data['steps'], data['values'],
                color=COLORS['baseline'], linewidth=2, label='Baseline', alpha=0.8)
    if 'train/top5_acc' in expert_metrics:
        data = expert_metrics['train/top5_acc']
        ax2.plot(data['steps'], data['values'],
                color=COLORS['expert'], linewidth=2, label='Expert', alpha=0.8)
    ax2.set_xlabel('Steps', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Top-5 Accuracy', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()

    # Bottom row: Final metrics
    ax3 = fig.add_subplot(gs[2, 0])
    metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
    labels = ['Top-1', 'Top-3', 'Top-5']
    x = np.arange(len(labels))
    width = 0.35

    baseline_values = [baseline_eval.get(m, 0) for m in metrics]
    expert_values = [expert_eval.get(m, 0) for m in metrics]

    ax3.bar(x - width/2, baseline_values, width,
            label='Baseline', color=COLORS['baseline'], alpha=0.8)
    ax3.bar(x + width/2, expert_values, width,
            label='Expert', color=COLORS['expert'], alpha=0.8)

    ax3.set_xlabel('Metric', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Final Accuracy (vs Ground Truth)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_ylim(0, 100)

    ax4 = fig.add_subplot(gs[2, 1])
    depths = [5, 10, 15]
    baseline_agreement = [baseline_eval.get(f'sf_agreement_depth{d}', 0) for d in depths]
    expert_agreement = [expert_eval.get(f'sf_agreement_depth{d}', 0) for d in depths]
    
    x = np.arange(len(depths))
    ax4.bar(x - width/2, baseline_agreement, width, label='Baseline',
            color=COLORS['baseline'], alpha=0.8)
    ax4.bar(x + width/2, expert_agreement, width, label='Expert',
            color=COLORS['expert'], alpha=0.8)
    
    ax4.set_xlabel('Stockfish Depth', fontsize=11)
    ax4.set_ylabel('Agreement (%)', fontsize=11)
    ax4.set_title('Stockfish Agreement', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{d}' for d in depths])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.set_ylim(0, 100)

    plt.suptitle('Chess Behavioral Cloning: Comprehensive Evaluation',
                fontsize=18, fontweight='bold', y=0.995)

    output_path = output_dir / 'final_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    return output_path


def main():
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE EVALUATION PLOTS")
    print("="*80)

    # Define paths
    base_dir = Path("/workspace/6s890-finalproject")

    baseline_logs = base_dir / "experiments/results/baseline_mixed_skill/logs"
    expert_logs = base_dir / "experiments/results/expert_LE22ct/logs"

    baseline_eval = base_dir / "results/evaluation/baseline_comprehensive_eval.json"
    expert_eval = base_dir / "results/evaluation/expert_comprehensive_eval.json"

    output_dir = base_dir / "results/evaluation/plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Extract metrics from TensorBoard logs
    print("\nExtracting TensorBoard metrics...")
    print("Baseline:")
    baseline_metrics = extract_metrics_from_logs(baseline_logs)
    print("Expert:")
    expert_metrics = extract_metrics_from_logs(expert_logs)

    # Analyze for target accuracy
    print("\nAnalyzing logs for sample complexity target...")
    baseline_max, baseline_final = analyze_logs_for_target(baseline_metrics)
    expert_max, expert_final = analyze_logs_for_target(expert_metrics)
    
    print(f"  Baseline Top-5: max={baseline_max:.2f}%, final={baseline_final:.2f}%")
    print(f"  Expert Top-5: max={expert_max:.2f}%, final={expert_final:.2f}%")
    
    target_accuracy = min(baseline_max, expert_max) * 0.85
    print(f"  → Using target: {target_accuracy:.1f}%")

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
    steps_to_target = None
    try:
        path, steps_to_target = plot_sample_complexity(baseline_metrics, expert_metrics, 
                                                       output_dir, target_accuracy)
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

    # 4. Stockfish alignment
    try:
        path = plot_stockfish_alignment(baseline_eval_data, expert_eval_data, output_dir)
        if path:
            saved_plots.append(path)
    except Exception as e:
        print(f"  Error creating Stockfish alignment plot: {e}")

    # 5. Final summary
    try:
        path = plot_final_summary(baseline_eval_data, expert_eval_data, 
                                 baseline_metrics, expert_metrics, output_dir, steps_to_target)
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
