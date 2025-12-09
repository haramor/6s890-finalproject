#!/usr/bin/env python3
"""
Auto-generated plotting script for training curves.

Usage:
    python plot_training_curves.py
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
with open('training_data_downsampled.json') as f:
    data = json.load(f)

# Get all unique metrics
all_metrics = set()
for exp_data in data.values():
    all_metrics.update(exp_data['metrics'].keys())

# Plot each metric
for metric in sorted(all_metrics):
    plt.figure(figsize=(10, 6))
    
    for exp_name, exp_data in data.items():
        if metric in exp_data['metrics']:
            metric_data = exp_data['metrics'][metric]
            plt.plot(
                metric_data['steps'],
                metric_data['values'],
                label=exp_name,
                color=exp_data['color'],
                linewidth=2,
                alpha=0.8
            )
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel(metric.split('/')[-1].replace('_', ' ').title(), fontsize=12)
    plt.title(f'{metric.split("/")[-1].replace("_", " ").title()} Over Training', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save
    filename = metric.replace('/', '_') + '.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

print("All plots saved!")
