"""
Comprehensive Training Diagnostics
Checks TensorBoard logs, checkpoints, and training health
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator

print("="*80)
print(" "*20 + "TRAINING DIAGNOSTICS REPORT")
print("="*80)
print()

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "─"*80)
    print(f"  {title}")
    print("─"*80)

def load_tensorboard_logs(log_dir):
    """Load all scalars from TensorBoard logs with detailed output."""
    print(f"\n📂 Searching for logs in: {log_dir}")
    
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"   ❌ Directory does not exist!")
        return {}
    
    # Find event files
    event_files = list(log_path.glob('events.out.tfevents.*'))
    if not event_files:
        print(f"   ❌ No TensorBoard event files found")
        return {}
    
    print(f"   ✓ Found {len(event_files)} event file(s)")
    for ef in event_files:
        print(f"     - {ef.name}")
    
    ea = event_accumulator.EventAccumulator(str(event_files[0]))
    ea.Reload()
    
    print(f"\n📊 Loading metrics...")
    data = {}
    scalar_tags = ea.Tags()['scalars']
    print(f"   Found {len(scalar_tags)} metric types")
    
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        df = pd.DataFrame([(e.step, e.value) for e in events], 
                          columns=['step', 'value'])
        data[tag] = df
        
        # Check for issues
        n_total = len(df)
        n_nan = df['value'].isna().sum()
        n_inf = np.isinf(df['value']).sum()
        n_valid = n_total - n_nan - n_inf
        
        status = "✓" if n_valid == n_total else "⚠"
        print(f"   {status} {tag}: {n_valid}/{n_total} valid values", end="")
        if n_nan > 0:
            print(f" ({n_nan} NaN)", end="")
        if n_inf > 0:
            print(f" ({n_inf} Inf)", end="")
        print()
    
    return data

def check_checkpoint(checkpoint_path):
    """Detailed checkpoint inspection."""
    print(f"\n📦 Checkpoint: {checkpoint_path.name}")
    print(f"   Path: {checkpoint_path}")
    print(f"   Size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"   Keys in checkpoint: {list(checkpoint.keys())}")
        
        # Check metadata
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'step' in checkpoint:
            print(f"   Step: {checkpoint['step']}")
        if 'val_acc' in checkpoint:
            print(f"   Validation Accuracy: {checkpoint['val_acc']:.4f}")
        
        # Check model state
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"   Model parameters: {len(state_dict)} tensors")
            
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
            
            # Check for NaN/Inf
            has_nan = False
            has_inf = False
            nan_params = []
            inf_params = []
            
            for name, param in state_dict.items():
                if torch.isnan(param).any():
                    has_nan = True
                    nan_params.append(name)
                if torch.isinf(param).any():
                    has_inf = True
                    inf_params.append(name)
            
            if has_nan:
                print(f"   ❌ NaN detected in {len(nan_params)} parameter(s):")
                for name in nan_params[:5]:  # Show first 5
                    print(f"      - {name}")
                if len(nan_params) > 5:
                    print(f"      ... and {len(nan_params)-5} more")
            
            if has_inf:
                print(f"   ❌ Inf detected in {len(inf_params)} parameter(s):")
                for name in inf_params[:5]:
                    print(f"      - {name}")
                if len(inf_params) > 5:
                    print(f"      ... and {len(inf_params)-5} more")
            
            if not has_nan and not has_inf:
                print(f"   ✓ Model weights are healthy (no NaN/Inf)")
                
                # Show weight statistics
                all_weights = torch.cat([p.flatten() for p in state_dict.values()])
                print(f"   Weight statistics:")
                print(f"      Min: {all_weights.min().item():.6f}")
                print(f"      Max: {all_weights.max().item():.6f}")
                print(f"      Mean: {all_weights.mean().item():.6f}")
                print(f"      Std: {all_weights.std().item():.6f}")
        
        # Check optimizer state
        if 'optimizer_state_dict' in checkpoint:
            opt_dict = checkpoint['optimizer_state_dict']
            print(f"   ✓ Optimizer state present")
            if 'param_groups' in opt_dict:
                for i, pg in enumerate(opt_dict['param_groups']):
                    if 'lr' in pg:
                        print(f"      Param group {i} LR: {pg['lr']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR loading checkpoint: {e}")
        return False

def analyze_training_metrics(logs, experiment_name):
    """Analyze and print training metrics statistics."""
    print_section(f"Training Metrics Analysis: {experiment_name}")
    
    if not logs:
        print("   ❌ No logs available")
        return {}
    
    stats = {}
    
    # Training metrics
    print("\n📈 TRAINING METRICS:")
    for metric in ['train/loss', 'train/ce_loss', 'train/top1_acc', 'train/top3_acc']:
        if metric in logs:
            df = logs[metric]
            valid_df = df[np.isfinite(df['value'])]
            
            if len(valid_df) > 0:
                stats[metric] = {
                    'first': valid_df['value'].iloc[0],
                    'last': valid_df['value'].iloc[-1],
                    'min': valid_df['value'].min(),
                    'max': valid_df['value'].max(),
                    'mean': valid_df['value'].mean(),
                    'std': valid_df['value'].std(),
                }
                
                print(f"\n   {metric.split('/')[1].upper()}:")
                print(f"      First value: {stats[metric]['first']:.4f}")
                print(f"      Last value:  {stats[metric]['last']:.4f}")
                print(f"      Min:         {stats[metric]['min']:.4f}")
                print(f"      Max:         {stats[metric]['max']:.4f}")
                print(f"      Mean:        {stats[metric]['mean']:.4f}")
                print(f"      Std:         {stats[metric]['std']:.4f}")
                
                # Check for improvement
                if 'acc' in metric:
                    improvement = stats[metric]['last'] - stats[metric]['first']
                    print(f"      Change:      {improvement:+.4f} ({'↑' if improvement > 0 else '↓'})")
                elif 'loss' in metric:
                    improvement = stats[metric]['first'] - stats[metric]['last']
                    print(f"      Change:      {improvement:+.4f} ({'↓' if improvement > 0 else '↑'})")
            else:
                print(f"\n   {metric}: ❌ All values are NaN/Inf")
    
    # Validation metrics
    print("\n📊 VALIDATION METRICS:")
    for metric in ['val/loss', 'val/top1_acc', 'val/top3_acc']:
        if metric in logs:
            df = logs[metric]
            valid_df = df[np.isfinite(df['value'])]
            
            if len(valid_df) > 0:
                stats[metric] = {
                    'best': valid_df['value'].max() if 'acc' in metric else valid_df['value'].min(),
                    'last': valid_df['value'].iloc[-1],
                    'mean': valid_df['value'].mean(),
                    'n_evals': len(valid_df),
                }
                
                print(f"\n   {metric.split('/')[1].upper()}:")
                print(f"      Best value:  {stats[metric]['best']:.4f}")
                print(f"      Last value:  {stats[metric]['last']:.4f}")
                print(f"      Mean:        {stats[metric]['mean']:.4f}")
                print(f"      # Evals:     {stats[metric]['n_evals']}")
            else:
                print(f"\n   {metric}: ❌ All values are NaN/Inf")
    
    # Entropy-specific metrics
    if 'train/entropy' in logs:
        print("\n🔀 ENTROPY METRICS:")
        df = logs['train/entropy']
        valid_df = df[np.isfinite(df['value'])]
        
        if len(valid_df) > 0:
            print(f"      First value: {valid_df['value'].iloc[0]:.4f}")
            print(f"      Last value:  {valid_df['value'].iloc[-1]:.4f}")
            print(f"      Mean:        {valid_df['value'].mean():.4f}")
            print(f"      Max entropy (uniform): {np.log(1971):.4f}")  # log(vocab_size)
    
    return stats

# ============================================================================
# MAIN DIAGNOSTICS
# ============================================================================

print_section("EXPERIMENT 1: Standard Training (expert_standard_2k)")

standard_results_dir = Path('/workspace/6s890-finalproject/experiments/results/expert_standard_2k')
print(f"\n📁 Results directory: {standard_results_dir}")
print(f"   Exists: {'✓' if standard_results_dir.exists() else '❌'}")

# Load logs
standard_logs = load_tensorboard_logs(standard_results_dir / 'logs')

# Analyze metrics
standard_stats = analyze_training_metrics(standard_logs, "Standard Training")

# Check checkpoints
print("\n💾 CHECKPOINTS:")
checkpoint_dir = standard_results_dir / 'checkpoints'
if checkpoint_dir.exists():
    checkpoints = sorted(checkpoint_dir.glob('*.pt'))
    print(f"   Found {len(checkpoints)} checkpoint(s)")
    for ckpt in checkpoints:
        check_checkpoint(ckpt)
else:
    print(f"   ❌ Checkpoint directory not found")

# ============================================================================

print_section("EXPERIMENT 2: Entropy Regularization (expert_entropy_2k)")

entropy_results_dir = Path('/workspace/6s890-finalproject/experiments/results/expert_entropy_2k')
print(f"\n📁 Results directory: {entropy_results_dir}")
print(f"   Exists: {'✓' if entropy_results_dir.exists() else '❌'}")

# Load logs
entropy_logs = load_tensorboard_logs(entropy_results_dir / 'logs')

# Analyze metrics
entropy_stats = analyze_training_metrics(entropy_logs, "Entropy Regularization")

# Check checkpoints
print("\n💾 CHECKPOINTS:")
checkpoint_dir = entropy_results_dir / 'checkpoints'
if checkpoint_dir.exists():
    checkpoints = sorted(checkpoint_dir.glob('*.pt'))
    print(f"   Found {len(checkpoints)} checkpoint(s)")
    for ckpt in checkpoints:
        check_checkpoint(ckpt)
else:
    print(f"   ❌ Checkpoint directory not found")

# ============================================================================

print_section("COMPARISON SUMMARY")

print("\n📊 FINAL RESULTS COMPARISON:")
print("\n" + " "*10 + "Standard" + " "*10 + "Entropy")
print("─"*80)

metrics_to_compare = [
    ('train/loss', 'Final Train Loss'),
    ('train/top1_acc', 'Final Train Top-1 Acc'),
    ('train/top3_acc', 'Final Train Top-3 Acc'),
    ('val/top1_acc', 'Best Val Top-1 Acc'),
    ('val/top3_acc', 'Best Val Top-3 Acc'),
]

for metric_key, metric_name in metrics_to_compare:
    std_val = "N/A"
    ent_val = "N/A"
    
    if metric_key in standard_stats:
        if 'last' in standard_stats[metric_key]:
            std_val = f"{standard_stats[metric_key]['last']:.4f}"
        elif 'best' in standard_stats[metric_key]:
            std_val = f"{standard_stats[metric_key]['best']:.4f}"
    
    if metric_key in entropy_stats:
        if 'last' in entropy_stats[metric_key]:
            ent_val = f"{entropy_stats[metric_key]['last']:.4f}"
        elif 'best' in entropy_stats[metric_key]:
            ent_val = f"{entropy_stats[metric_key]['best']:.4f}"
    
    print(f"{metric_name:30s} {std_val:>12s}    {ent_val:>12s}")

print("\n" + "="*80)
print(" "*20 + "DIAGNOSTICS COMPLETE")
print("="*80)
print()
print("Next steps:")
print("  1. If all values are NaN/0, training failed - check training logs")
print("  2. If values look good, proceed to create plots")
print("  3. Run: python scripts/create_training_plots.py")
print()
