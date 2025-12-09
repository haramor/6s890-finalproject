"""
Sample Complexity Analysis - Fixed for decimal accuracy values
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard", "--break-system-packages"])
    from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_logs(log_dir):
    """Load all metrics from TensorBoard logs."""
    log_dir = Path(log_dir)
    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    
    if not event_files:
        return {}
    
    all_metrics = {}
    
    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(
                str(event_file),
                size_guidance={event_accumulator.SCALARS: 0}
            )
            ea.Reload()
            
            for tag in ea.Tags()['scalars']:
                if tag not in all_metrics:
                    all_metrics[tag] = {'steps': [], 'values': []}
                
                events = ea.Scalars(tag)
                for event in events:
                    all_metrics[tag]['steps'].append(event.step)
                    all_metrics[tag]['values'].append(event.value)
        except Exception as e:
            print(f"Error loading {event_file.name}: {e}")
    
    # Remove duplicates and sort
    for tag in all_metrics:
        step_dict = {}
        for s, v in zip(all_metrics[tag]['steps'], all_metrics[tag]['values']):
            step_dict[s] = v
        
        sorted_items = sorted(step_dict.items())
        all_metrics[tag]['steps'] = [s for s, v in sorted_items]
        all_metrics[tag]['values'] = [v for s, v in sorted_items]
    
    return all_metrics


def steps_to_samples(steps, batch_size, batches_per_step):
    """Convert training steps to number of samples seen."""
    samples_per_step = batch_size * batches_per_step
    return [step * samples_per_step for step in steps]


def find_samples_to_target(samples, values, target_value):
    """Find number of training samples needed to reach target performance."""
    for s, v in zip(samples, values):
        if v >= target_value:
            return s
    return None


def plot_learning_curves(results, output_dir, target_value=0.30):
    """Plot learning curves comparing baseline vs expert."""
    output_dir = Path(output_dir)
    
    # Training metrics (convert to percentage for display)
    metrics_to_plot = [
        ('train/top1_acc', 'Training Top-1 Accuracy (%)', True),
        ('train/top3_acc', 'Training Top-3 Accuracy (%)', True),
        ('val/top1_acc', 'Validation Top-1 Accuracy (%)', True),
        ('val/top3_acc', 'Validation Top-3 Accuracy (%)', True),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sample Complexity: Baseline vs Expert', fontsize=16)
    
    for idx, (metric, ylabel, convert_to_pct) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        for exp_name in ['baseline', 'expert']:
            if exp_name not in results or metric not in results[exp_name]['metrics']:
                continue
            
            data = results[exp_name]
            metric_data = data['metrics'][metric]
            
            steps = metric_data['steps']
            values = metric_data['values']
            
            # Convert to percentage if needed
            if convert_to_pct:
                values = [v * 100 for v in values]
            
            samples = steps_to_samples(steps, data['batch_size'], data['batches_per_step'])
            samples_m = [s / 1e6 for s in samples]
            
            ax.plot(samples_m, values, label=exp_name.capitalize(), 
                   linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Training Samples (millions)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add target line for top-3 training accuracy
        if metric == 'train/top3_acc':
            ax.axhline(y=target_value * 100, color='r', linestyle='--', 
                      label=f'Target: {target_value*100:.0f}%', linewidth=1.5)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_complexity_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'sample_complexity_curves.png'}")
    plt.close()


def main():
    print("="*70)
    print("SAMPLE COMPLEXITY ANALYSIS")
    print("="*70)
    
    experiments = {
        'baseline': {
            'logs': '/workspace/6s890-finalproject/experiments/results/baseline_mixed_skill/logs',
            'batch_size': 64,
            'batches_per_step': 4,
        },
        'expert': {
            'logs': '/workspace/6s890-finalproject/experiments/results/expert_LE22ct/logs',
            'batch_size': 64,
            'batches_per_step': 4,
        }
    }
    
    output_dir = Path('/workspace/6s890-finalproject/results/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for exp_name, exp_config in experiments.items():
        print(f"\n{exp_name.upper()}:")
        
        metrics = load_tensorboard_logs(exp_config['logs'])
        
        if not metrics:
            print(f"  ERROR: No metrics found")
            continue
        
        # Print key statistics
        if 'train/top3_acc' in metrics:
            steps = metrics['train/top3_acc']['steps']
            values = metrics['train/top3_acc']['values']
            print(f"  Training steps: {steps[0]} to {steps[-1]}")
            print(f"  Training samples: {steps[-1] * exp_config['batch_size'] * exp_config['batches_per_step']:,}")
            print(f"  Max train/top3_acc: {max(values)*100:.2f}%")
        
        if 'val/top3_acc' in metrics:
            values = metrics['val/top3_acc']['values']
            print(f"  Max val/top3_acc: {max(values)*100:.2f}%")
        
        all_results[exp_name] = {
            'batch_size': exp_config['batch_size'],
            'batches_per_step': exp_config['batches_per_step'],
            'metrics': metrics,
        }
    
    # Analysis
    print(f"\n{'='*70}")
    print("SAMPLE EFFICIENCY ANALYSIS")
    print(f"{'='*70}")
    
    target_metric = 'train/top3_acc'
    target_value = 0.30  # 30% in decimal form
    
    print(f"\nTarget: {target_value*100:.0f}% on {target_metric}")
    
    efficiency_results = {}
    
    for exp_name, data in all_results.items():
        if target_metric not in data['metrics']:
            continue
        
        metric_data = data['metrics'][target_metric]
        steps = metric_data['steps']
        values = metric_data['values']
        samples = steps_to_samples(steps, data['batch_size'], data['batches_per_step'])
        
        samples_to_target = find_samples_to_target(samples, values, target_value)
        max_value = max(values) if values else 0
        final_value = values[-1] if values else 0
        max_samples = samples[-1] if samples else 0
        
        efficiency_results[exp_name] = {
            'samples_to_target': samples_to_target,
            'max_value': max_value * 100,  # Convert to percentage
            'final_value': final_value * 100,
            'max_samples': max_samples,
            'reached_target': samples_to_target is not None,
        }
        
        print(f"\n{exp_name.upper()}:")
        print(f"  Total samples: {max_samples:,}")
        print(f"  Final {target_metric}: {final_value*100:.2f}%")
        print(f"  Max {target_metric}: {max_value*100:.2f}%")
        
        if samples_to_target:
            print(f"  ✓ Reached {target_value*100:.0f}% at {samples_to_target:,} samples")
        else:
            print(f"  ✗ Never reached {target_value*100:.0f}%")
    
    # Compare
    print(f"\n{'='*70}")
    print("HYPOTHESIS TEST")
    print(f"{'='*70}")
    print("\nHypothesis: Expert data requires fewer samples to reach target performance")
    
    baseline = efficiency_results.get('baseline', {})
    expert = efficiency_results.get('expert', {})
    
    baseline_samples = baseline.get('samples_to_target', None)
    expert_samples = expert.get('samples_to_target', None)
    
    if baseline_samples and expert_samples:
        improvement = ((baseline_samples - expert_samples) / baseline_samples) * 100
        
        print(f"\nBaseline: {baseline_samples:,} samples to reach {target_value*100:.0f}%")
        print(f"Expert:   {expert_samples:,} samples to reach {target_value*100:.0f}%")
        print(f"\nSample efficiency improvement: {improvement:+.1f}%")
        
        if improvement > 5:  # 5% threshold
            print(f"\n✓ HYPOTHESIS SUPPORTED")
            print(f"  Expert requires {improvement:.1f}% fewer samples to reach target")
        elif improvement < -5:
            print(f"\n✗ HYPOTHESIS REJECTED")
            print(f"  Expert requires {-improvement:.1f}% MORE samples to reach target")
        else:
            print(f"\n≈ HYPOTHESIS INCONCLUSIVE")
            print(f"  Difference is small ({abs(improvement):.1f}%), models perform similarly")
    else:
        print(f"\n⚠ Neither model reached {target_value*100:.0f}% - both undertrained")
        
        if baseline and expert:
            baseline_max = baseline.get('max_value', 0)
            expert_max = expert.get('max_value', 0)
            baseline_samples_used = baseline.get('max_samples', 0)
            expert_samples_used = expert.get('max_samples', 0)
            
            print(f"\nBest performance achieved:")
            print(f"  Baseline: {baseline_max:.2f}% at {baseline_samples_used:,} samples")
            print(f"  Expert:   {expert_max:.2f}% at {expert_samples_used:,} samples")
            
            perf_diff = expert_max - baseline_max
            
            if abs(perf_diff) < 1.0:
                print(f"\n  → Performance is essentially identical ({perf_diff:+.2f}pp)")
            elif perf_diff > 0:
                print(f"\n  → Expert is {perf_diff:.2f}pp better")
            else:
                print(f"\n  → Baseline is {-perf_diff:.2f}pp better")
            
            # Sample efficiency
            if baseline_samples_used > 0 and expert_samples_used > 0:
                baseline_eff = baseline_max / (baseline_samples_used / 1e6)
                expert_eff = expert_max / (expert_samples_used / 1e6)
                
                print(f"\n  Sample efficiency (% accuracy per 1M samples):")
                print(f"    Baseline: {baseline_eff:.3f}%")
                print(f"    Expert:   {expert_eff:.3f}%")
                
                if expert_eff > baseline_eff * 1.1:
                    ratio = expert_eff / baseline_eff
                    print(f"    → Expert is {ratio:.2f}x more sample-efficient")
                    print(f"\n✓ PARTIAL SUPPORT: Expert learns faster per sample")
                elif baseline_eff > expert_eff * 1.1:
                    ratio = baseline_eff / expert_eff
                    print(f"    → Baseline is {ratio:.2f}x more sample-efficient")
                    print(f"\n✗ HYPOTHESIS REJECTED: Baseline learns faster per sample")
                else:
                    print(f"    → Similar sample efficiency")
                    print(f"\n≈ NO DIFFERENCE: Both models learn at similar rates")
    
    # Save
    output_file = output_dir / 'sample_complexity_final.json'
    with open(output_file, 'w') as f:
        json.dump({
            'experiments': {
                name: {
                    'batch_size': data['batch_size'],
                    'batches_per_step': data['batches_per_step'],
                }
                for name, data in all_results.items()
            },
            'efficiency_results': efficiency_results,
            'target_metric': target_metric,
            'target_value_pct': target_value * 100,
        }, f, indent=2)
    print(f"\n✓ Saved: {output_file}")
    
    # Plot
    try:
        plot_learning_curves(all_results, output_dir, target_value)
    except Exception as e:
        print(f"Plot error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print("\nKey files to download:")
    print("  - sample_complexity_curves.png")
    print("  - sample_complexity_final.json")


if __name__ == '__main__':
    main()
