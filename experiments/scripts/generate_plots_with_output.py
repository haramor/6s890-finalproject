"""
Generate Plots for Low ELO Evaluation Results + Terminal Output
"""

import sys
from pathlib import Path
import json

# Install dependencies
for pkg in ['matplotlib', 'numpy']:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages"])

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.titlesize'] = 16

COLORS = {
    'baseline': '#2E86AB',
    'expert': '#A23B72',
    'game_theoretic_qre': '#F18F01'
}


def load_results(eval_dir):
    """Load evaluation results."""
    results = {}
    for json_file in eval_dir.glob('*_eval.json'):
        exp_name = json_file.stem.replace('_eval', '')
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Filter out game_theoretic if it failed
                if data.get('legal_move_rate', 0) < 50:
                    print(f"  ⚠️  Skipping {exp_name} (legal move rate too low: {data.get('legal_move_rate', 0):.1f}%)")
                    continue
                results[exp_name] = data
                print(f"  ✓ Loaded: {exp_name}")
        except Exception as e:
            print(f"  ✗ Error: {json_file}: {e}")
    return results


def print_results_table(results):
    """Print results in a nice table format."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS TABLE")
    print("="*80)
    
    exp_names = list(results.keys())
    
    # Header
    print(f"\n{'Metric':<40} " + " ".join([f"{name.replace('_', ' ').title():>15}" for name in exp_names]))
    print("-" * 80)
    
    # Rows
    metrics = [
        ('n_evaluated', 'Samples Evaluated', lambda x: f"{int(x):,}"),
        ('legal_move_rate', 'Legal Move Rate (%)', lambda x: f"{x:.2f}%"),
        ('top1_accuracy', 'Top-1 Accuracy (%)', lambda x: f"{x:.2f}%"),
        ('top3_accuracy', 'Top-3 Accuracy (%)', lambda x: f"{x:.2f}%"),
        ('top5_accuracy', 'Top-5 Accuracy (%)', lambda x: f"{x:.2f}%"),
    ]
    
    for key, label, fmt in metrics:
        values = [fmt(results[e].get(key, 0)) for e in exp_names]
        print(f"{label:<40} " + " ".join([f"{v:>15}" for v in values]))
    
    print("\n" + "-" * 80)
    print("STOCKFISH ALIGNMENT")
    print("-" * 80)
    
    for depth in [5, 8, 10]:
        elo = {5: "~2000", 8: "~2200", 10: "~2400"}[depth]
        
        # Exact agreement
        key = f'sf_agreement_depth{depth}'
        values = [f"{results[e].get(key, 0):.2f}%" for e in exp_names]
        print(f"SF Agreement D{depth} ({elo} ELO)".ljust(40) + " ".join([f"{v:>15}" for v in values]))
        
        # Model in SF Top-5
        key = f'model_in_sf_top5_depth{depth}'
        values = [f"{results[e].get(key, 0):.2f}%" for e in exp_names]
        print(f"  → Model in SF Top-5 D{depth}".ljust(40) + " ".join([f"{v:>15}" for v in values]))
        
        # SF in Model Top-5
        key = f'sf_in_model_top5_depth{depth}'
        values = [f"{results[e].get(key, 0):.2f}%" for e in exp_names]
        print(f"  → SF in Model Top-5 D{depth}".ljust(40) + " ".join([f"{v:>15}" for v in values]))
        print()
    
    # Improvements
    if len(exp_names) == 2:
        print("-" * 80)
        print("IMPROVEMENTS (Expert vs Baseline)")
        print("-" * 80)
        
        baseline = results[exp_names[0]]
        expert = results[exp_names[1]]
        
        improvements = [
            ('top1_accuracy', 'Top-1 Accuracy'),
            ('top5_accuracy', 'Top-5 Accuracy'),
            ('sf_agreement_depth5', 'SF Agreement D5'),
            ('sf_agreement_depth8', 'SF Agreement D8'),
            ('sf_agreement_depth10', 'SF Agreement D10'),
        ]
        
        for key, label in improvements:
            base_val = baseline.get(key, 0)
            exp_val = expert.get(key, 0)
            diff = exp_val - base_val
            pct_change = (diff / base_val * 100) if base_val > 0 else 0
            
            sign = "+" if diff >= 0 else ""
            print(f"{label:<40} {sign}{diff:.2f}% ({sign}{pct_change:.1f}% relative)")
    
    print("\n" + "="*80 + "\n")


def plot_accuracy_comparison(results, output_dir):
    """Bar chart of accuracies."""
    print("\nCreating accuracy comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
    labels = ['Top-1', 'Top-3', 'Top-5']
    x = np.arange(len(labels))
    
    exp_names = list(results.keys())
    width = 0.35 if len(exp_names) == 2 else 0.25
    
    for i, exp_name in enumerate(exp_names):
        data = results[exp_name]
        values = [data.get(m, 0) for m in metrics]
        color = COLORS.get(exp_name, '#333333')
        label = exp_name.replace('_', ' ').title()
        
        offset = (i - len(exp_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Accuracy Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison\n(vs Ground Truth Moves)', 
                 fontsize=14, fontweight='bold')
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


def plot_stockfish_alignment(results, output_dir):
    """Stockfish alignment at different depths."""
    print("\nCreating Stockfish alignment plot...")
    
    depths = [5, 8, 10]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Stockfish Alignment by Depth\n(Club to Master Level)', 
                 fontsize=16, fontweight='bold')
    
    exp_names = list(results.keys())
    x = np.arange(len(depths))
    width = 0.35 if len(exp_names) == 2 else 0.25
    
    # Plot 1: Exact Agreement
    ax = axes[0]
    for i, exp_name in enumerate(exp_names):
        data = results[exp_name]
        agreement = [data.get(f'sf_agreement_depth{d}', 0) for d in depths]
        color = COLORS.get(exp_name, '#333333')
        label = exp_name.replace('_', ' ').title()
        
        offset = (i - len(exp_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, agreement, width, label=label, color=color, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Stockfish Depth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Agreement (%)', fontsize=11, fontweight='bold')
    ax.set_title('Exact Agreement', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}\n(~{["2000", "2200", "2400"][i]} ELO)' for i, d in enumerate(depths)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 50)
    
    # Plot 2: Model in SF Top-5
    ax = axes[1]
    for i, exp_name in enumerate(exp_names):
        data = results[exp_name]
        in_sf = [data.get(f'model_in_sf_top5_depth{d}', 0) for d in depths]
        color = COLORS.get(exp_name, '#333333')
        label = exp_name.replace('_', ' ').title()
        
        offset = (i - len(exp_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, in_sf, width, label=label, color=color, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Stockfish Depth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Model in SF Top-5', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}\n(~{["2000", "2200", "2400"][i]} ELO)' for i, d in enumerate(depths)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 100)
    
    # Plot 3: SF in Model Top-5
    ax = axes[2]
    for i, exp_name in enumerate(exp_names):
        data = results[exp_name]
        sf_in = [data.get(f'sf_in_model_top5_depth{d}', 0) for d in depths]
        color = COLORS.get(exp_name, '#333333')
        label = exp_name.replace('_', ' ').title()
        
        offset = (i - len(exp_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, sf_in, width, label=label, color=color, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Stockfish Depth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('SF in Model Top-5', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}\n(~{["2000", "2200", "2400"][i]} ELO)' for i, d in enumerate(depths)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output_path = output_dir / 'stockfish_alignment.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    return output_path


def plot_summary_table(results, output_dir):
    """Summary table."""
    print("\nCreating summary table...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    exp_names = list(results.keys())
    
    table_data = [
        ['Metric'] + [e.replace('_', ' ').title() for e in exp_names]
    ]
    
    rows = [
        ('n_evaluated', 'Samples Evaluated', lambda x: f"{int(x):,}"),
        ('legal_move_rate', 'Legal Move Rate (%)', lambda x: f"{x:.2f}"),
        ('top1_accuracy', 'Top-1 Accuracy (%)', lambda x: f"{x:.2f}"),
        ('top3_accuracy', 'Top-3 Accuracy (%)', lambda x: f"{x:.2f}"),
        ('top5_accuracy', 'Top-5 Accuracy (%)', lambda x: f"{x:.2f}"),
        ('sf_agreement_depth5', 'SF Agreement D5 (~2000 ELO)', lambda x: f"{x:.2f}"),
        ('sf_agreement_depth8', 'SF Agreement D8 (~2200 ELO)', lambda x: f"{x:.2f}"),
        ('sf_agreement_depth10', 'SF Agreement D10 (~2400 ELO)', lambda x: f"{x:.2f}"),
    ]
    
    for key, label, fmt in rows:
        row = [label]
        for exp_name in exp_names:
            value = results[exp_name].get(key, 0)
            row.append(fmt(value))
        table_data.append(row)
    
    # Add improvement row
    if len(exp_names) == 2:
        baseline_name = exp_names[0]
        expert_name = exp_names[1]
        
        improvements = ['Δ (Expert - Baseline)', '']
        for key, _, _ in rows[2:]:  # Skip samples and legal rate
            baseline_val = results[baseline_name].get(key, 0)
            expert_val = results[expert_name].get(key, 0)
            diff = expert_val - baseline_val
            improvements.append(f"{diff:+.2f}")
        
        table_data.append(improvements)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.45] + [0.55/len(exp_names)]*len(exp_names))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(exp_names) + 1):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight improvement row
    if len(table_data) > len(rows) + 1:
        for i in range(len(exp_names) + 1):
            table[(len(table_data) - 1, i)].set_facecolor('#FFD700')
            table[(len(table_data) - 1, i)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data) - (1 if len(exp_names) == 2 else 0)):
        for j in range(len(exp_names) + 1):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    ax.set_title('Comprehensive Evaluation Results\n(Lower ELO Stockfish: Club to Master Level)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'summary_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    return output_path


def main():
    print("\n" + "="*80)
    print("GENERATING PLOTS FOR LOW ELO EVALUATION")
    print("="*80)
    
    eval_dir = Path("/workspace/6s890-finalproject/experiments/scripts/eval_results_low_elo")
    
    if not eval_dir.exists():
        print(f"ERROR: {eval_dir} not found")
        return
    
    output_dir = eval_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading results from: {eval_dir}")
    results = load_results(eval_dir)
    
    if not results:
        print("ERROR: No valid results found")
        return
    
    # Print results table first
    print_results_table(results)
    
    print(f"\nGenerating plots to: {output_dir}")
    
    saved = []
    
    try:
        saved.append(plot_accuracy_comparison(results, output_dir))
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    try:
        saved.append(plot_stockfish_alignment(results, output_dir))
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    try:
        saved.append(plot_summary_table(results, output_dir))
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n" + "="*80)
    print(f"COMPLETE - Generated {len(saved)} plots")
    print("="*80)
    for path in saved:
        print(f"  ✓ {path}")
    print(f"\nPlots saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
