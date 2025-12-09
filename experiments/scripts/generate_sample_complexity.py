"""
Generate Sample Complexity Curves - FIXED to handle fraction vs percentage
"""

import sys
from pathlib import Path

for pkg in ['tensorboard', 'matplotlib', 'numpy']:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages"])

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.titlesize'] = 16

COLORS = {
    'baseline': '#2E86AB',
    'expert': '#A23B72',
    'game_theoretic_qre': '#F18F01'
}


def extract_training_curves(log_dir, metric='train/top1_acc'):
    """Extract training curves from TensorBoard logs."""
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        return None, None
    
    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None, None
    
    all_steps = []
    all_values = []
    
    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()
            
            if metric not in ea.Tags()['scalars']:
                continue
            
            events = ea.Scalars(metric)
            for event in events:
                all_steps.append(event.step)
                all_values.append(event.value)
        except Exception as e:
            continue
    
    if not all_steps:
        return None, None
    
    # Remove duplicates and sort
    step_value_dict = {}
    for s, v in zip(all_steps, all_values):
        step_value_dict[s] = v
    
    sorted_steps = sorted(step_value_dict.keys())
    sorted_values = [step_value_dict[s] for s in sorted_steps]
    
    # CRITICAL FIX: Check if values are fractions (0-1) and convert to percentages
    max_val = max(sorted_values)
    if max_val < 1.5:  # If max is less than 1.5, it's a fraction not a percentage
        print(f"    Converting from fraction to percentage (max was {max_val:.4f})")
        sorted_values = [v * 100 for v in sorted_values]
    
    return sorted_steps, sorted_values


def print_training_summary(log_dirs):
    """Print training metrics summary."""
    print("\n" + "="*80)
    print("TRAINING METRICS SUMMARY")
    print("="*80)
    
    for exp_name, log_dir in log_dirs.items():
        print(f"\n{exp_name.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        # Top-1 Accuracy
        steps, values = extract_training_curves(log_dir, 'train/top1_acc')
        if steps and values:
            print(f"  Top-1 Accuracy:")
            print(f"    Initial: {values[0]:.2f}%")
            print(f"    Final: {values[-1]:.2f}%")
            print(f"    Max: {max(values):.2f}%")
            print(f"    Total Steps: {steps[-1]:,}")
        else:
            print(f"  Top-1 Accuracy: No data found")
        
        # Loss
        steps, values = extract_training_curves(log_dir, 'train/loss')
        if steps and values:
            print(f"  Training Loss:")
            print(f"    Initial: {values[0]:.4f}")
            print(f"    Final: {values[-1]:.4f}")
            print(f"    Min: {min(values):.4f}")
        else:
            print(f"  Training Loss: No data found")


def plot_sample_complexity_top1(log_dirs, output_dir):
    """Sample complexity for Top-1 accuracy."""
    print("\n" + "="*80)
    print("CREATING TOP-1 SAMPLE COMPLEXITY CURVE")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    all_data = {}
    max_accuracies = []
    
    # Extract data
    print("\nExtracting training data...")
    for exp_name, log_dir in log_dirs.items():
        steps, values = extract_training_curves(log_dir, 'train/top1_acc')
        if steps and values:
            all_data[exp_name] = {'steps': steps, 'values': values}
            max_accuracies.append(max(values))
            print(f"  {exp_name}: {len(steps)} points, max={max(values):.2f}%")
    
    if not all_data:
        print("  ERROR: No training data found")
        return None
    
    # Determine target (90% of minimum max)
    target = min(max_accuracies) * 0.90
    print(f"\n  Target accuracy: {target:.1f}%")
    print(f"  (90% of minimum max: {min(max_accuracies):.1f}%)")
    
    # Plot curves and find target steps
    steps_to_target = {}
    print(f"\n  Steps to reach {target:.1f}%:")
    
    for exp_name, data in all_data.items():
        color = COLORS.get(exp_name, '#333333')
        label = exp_name.replace('_', ' ').title()
        
        ax.plot(data['steps'], data['values'],
                color=color, linewidth=2.5, label=label, alpha=0.8, marker='o', markersize=2)
        
        # Find step where target is reached
        for step, val in zip(data['steps'], data['values']):
            if val >= target:
                steps_to_target[exp_name] = step
                ax.axvline(x=step, color=color, linestyle=':', linewidth=2, alpha=0.6)
                ax.plot(step, val, 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=2)
                print(f"    {exp_name.replace('_', ' ').title()}: {step:,} steps")
                break
        
        if exp_name not in steps_to_target:
            print(f"    {exp_name.replace('_', ' ').title()}: Never reached target (max={max(data['values']):.2f}%)")
    
    # Calculate efficiency
    if len(steps_to_target) >= 2:
        models = sorted(steps_to_target.keys())
        steps_1 = steps_to_target[models[0]]
        steps_2 = steps_to_target[models[1]]
        reduction = ((steps_1 - steps_2) / steps_1) * 100
        
        print(f"\n  EFFICIENCY COMPARISON:")
        if reduction > 0:
            print(f"    {models[1].replace('_', ' ').title()} is {reduction:.1f}% more efficient!")
            print(f"    ({steps_1 - steps_2:,} fewer steps needed)")
        else:
            print(f"    {models[0].replace('_', ' ').title()} is {abs(reduction):.1f}% more efficient!")
            print(f"    ({steps_2 - steps_1:,} fewer steps needed)")
    
    # Target line
    ax.axhline(y=target, color='#E63946', linestyle='--', linewidth=2.5,
               alpha=0.7, label=f'Target ({target:.1f}%)', zorder=0)
    
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Sample Complexity: Steps to Reach {target:.1f}% Top-1 Accuracy',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    
    # Annotation box
    if steps_to_target:
        text_lines = ["Steps to Target:"]
        for model in sorted(steps_to_target.keys()):
            steps = steps_to_target[model]
            text_lines.append(f"{model.replace('_', ' ').title()}: {steps:,}")
        
        if len(steps_to_target) >= 2:
            models = sorted(steps_to_target.keys())
            reduction = ((steps_to_target[models[0]] - steps_to_target[models[1]]) / steps_to_target[models[0]]) * 100
            text_lines.append("")
            if reduction > 0:
                text_lines.append(f"{models[1].replace('_', ' ').title()}: {reduction:.1f}% faster")
            else:
                text_lines.append(f"{models[0].replace('_', ' ').title()}: {abs(reduction):.1f}% faster")
        
        textstr = '\n'.join(text_lines)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    output_path = output_dir / 'sample_complexity_top1.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Saved: {output_path}")
    return output_path


def plot_training_curves(log_dirs, output_dir):
    """Full training curves comparison."""
    print("\nCreating training curves comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress Comparison', fontsize=16, fontweight='bold')
    
    metrics = [
        ('train/loss', 'Training Loss', axes[0, 0], False),
        ('train/top1_acc', 'Top-1 Accuracy (%)', axes[0, 1], True),
        ('train/top3_acc', 'Top-3 Accuracy (%)', axes[1, 0], True),
        ('train/lr', 'Learning Rate', axes[1, 1], False)
    ]
    
    for metric, title, ax, is_acc in metrics:
        plotted = False
        
        for exp_name, log_dir in log_dirs.items():
            steps, values = extract_training_curves(log_dir, metric)
            if steps and values:
                color = COLORS.get(exp_name, '#333333')
                label = exp_name.replace('_', ' ').title()
                ax.plot(steps, values, color=color, linewidth=2, label=label, alpha=0.8)
                plotted = True
        
        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        if plotted:
            ax.legend()
        
        if is_acc:
            ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    return output_path


def main():
    print("\n" + "="*80)
    print("SAMPLE COMPLEXITY CURVES (FIXED FOR FRACTION/PERCENTAGE)")
    print("="*80)
    
    base_dir = Path("/workspace/6s890-finalproject")
    
    log_dirs = {
        'baseline': base_dir / "experiments/results/baseline_mixed_skill/logs",
        'expert': base_dir / "experiments/results/expert_LE22ct/logs",
        'game_theoretic_qre': base_dir / "experiments/results/game_theoretic_qre/logs"
    }
    
    # Filter to existing
    existing_logs = {}
    print("\nChecking for training logs...")
    for name, path in log_dirs.items():
        if path.exists():
            existing_logs[name] = path
            print(f"  ✓ Found: {name}")
        else:
            print(f"  ✗ Missing: {name}")
    
    if not existing_logs:
        print("\nERROR: No training logs found")
        return
    
    # Print training summary
    print_training_summary(existing_logs)
    
    output_dir = base_dir / "experiments/scripts/eval_results_low_elo/plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    saved = []
    
    try:
        path = plot_sample_complexity_top1(existing_logs, output_dir)
        if path:
            saved.append(path)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        path = plot_training_curves(existing_logs, output_dir)
        if path:
            saved.append(path)
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n" + "="*80)
    print(f"COMPLETE - Generated {len(saved)} plots")
    print("="*80)
    for path in saved:
        print(f"  ✓ {path}")
    print(f"\nAll plots saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
