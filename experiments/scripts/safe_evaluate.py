"""
Safe Evaluation Script - Top-K Accuracy Only

This is a minimal evaluation script that ONLY runs top-k accuracy,
which is guaranteed to work. Use this if you need results quickly
and don't have time to implement the missing methods.

Usage:
    python safe_evaluate.py --checkpoint path/to/checkpoint.pt --config path/to/config.py
"""

import sys
import os
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add chess-transformers to path
chess_transformers_path = Path(__file__).parent.parent.parent / "chess-transformers"
sys.path.insert(0, str(chess_transformers_path))

try:
    from chess_transformers.transformers.models import ChessTransformerEncoder
    from chess_transformers.train.datasets import ChessDataset
except ImportError as e:
    print(f"Warning: Could not import from chess-transformers: {e}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str):
    """Load configuration from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def load_checkpoint(checkpoint_path: str, model: nn.Module) -> Dict:
    """Load model checkpoint and return metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'step': checkpoint.get('step', 'unknown'),
        }
    else:
        model.load_state_dict(checkpoint)
        metadata = {}

    print(f"Loaded checkpoint from: {checkpoint_path}")
    if metadata:
        print(f"Checkpoint metadata: {metadata}")

    return metadata


def evaluate_top_k_accuracy_safe(
    model: nn.Module,
    dataloader: DataLoader,
    k_values: List[int] = [1, 3, 5],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = None,
    save_frequency: int = 100
) -> Dict[str, float]:
    """
    Evaluate top-k accuracy with intermediate saving.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        k_values: List of k values
        device: Device for inference
        save_path: Path to save intermediate results (optional)
        save_frequency: Save every N batches

    Returns:
        Dictionary with top-k accuracies
    """
    print(f"\nEvaluating top-k accuracy (k={k_values})...")
    print(f"Device: {device}")
    print(f"Total batches: {len(dataloader)}")

    model.eval()
    model.to(device)

    # We'll compute accuracy batch by batch to avoid memory issues
    total_samples = 0
    k_correct = {k: 0 for k in k_values}

    intermediate_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Get predictions
            try:
                predictions = model(batch)  # (N, n_moves, vocab_size)

                # Extract first move predictions and targets
                pred_first_move = predictions[:, 0, :]  # (N, vocab_size)
                target_first_move = batch["moves"][:, 1]  # (N,)

                batch_size = pred_first_move.shape[0]

                # Compute top-k for this batch
                for k in k_values:
                    top_k_pred = torch.topk(pred_first_move, k=k, dim=1)[1]
                    correct = (top_k_pred == target_first_move.unsqueeze(1)).any(dim=1)
                    k_correct[k] += correct.sum().item()

                total_samples += batch_size

            except Exception as e:
                print(f"\nWarning: Error in batch {batch_idx}: {e}")
                continue

            # Intermediate saving
            if save_path and (batch_idx + 1) % save_frequency == 0:
                current_results = {
                    f"top_{k}": (k_correct[k] / total_samples * 100) if total_samples > 0 else 0
                    for k in k_values
                }
                current_results['samples_processed'] = total_samples
                current_results['batches_processed'] = batch_idx + 1

                intermediate_path = f"{save_path}.intermediate"
                with open(intermediate_path, 'w') as f:
                    json.dump(current_results, f, indent=2)

                print(f"\n[Batch {batch_idx + 1}/{len(dataloader)}] Saved intermediate results to {intermediate_path}")
                for k in k_values:
                    print(f"  Top-{k}: {current_results[f'top_{k}']:.2f}%")

    # Compute final accuracies
    if total_samples == 0:
        print("Warning: No samples were successfully processed!")
        return {f"top_{k}": 0.0 for k in k_values}

    results = {
        f"top_{k}": (k_correct[k] / total_samples * 100)
        for k in k_values
    }
    results['total_samples'] = total_samples

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    for k in k_values:
        print(f"  Top-{k} accuracy: {results[f'top_{k}']:.2f}%")
    print(f"  Total samples: {total_samples}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Safe evaluation (top-k accuracy only)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default="results/evaluation_safe",
                       help="Output directory")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (override config)")
    parser.add_argument("--save-frequency", type=int, default=50,
                       help="Save intermediate results every N batches")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"\n{'='*70}")
    print(f"Evaluating: {config.NAME}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*70}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create test dataloader
    try:
        batch_size = args.batch_size if args.batch_size else config.BATCH_SIZE

        test_loader = DataLoader(
            ChessDataset(
                data_folder=config.DATA_FOLDER,
                h5_file=config.H5_FILE,
                split="val",
                n_moves=config.N_MOVES
            ),
            batch_size=batch_size,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
            pin_memory=False
        )
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Batch size: {batch_size}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return 1

    # Load model
    try:
        model = ChessTransformerEncoder(config)
        print(f"Model created successfully")
    except Exception as e:
        print(f"Error creating model: {e}")
        return 1

    # Load checkpoint
    load_checkpoint(args.checkpoint, model)

    # Prepare save path
    save_path = output_dir / f"{config.NAME}_evaluation.json"

    # Run evaluation
    results = evaluate_top_k_accuracy_safe(
        model=model,
        dataloader=test_loader,
        k_values=[1, 3, 5],
        device=DEVICE,
        save_path=str(save_path),
        save_frequency=args.save_frequency
    )

    # Add metadata
    results['config_name'] = config.NAME
    results['experiment_type'] = config.EXPERIMENT_TYPE
    results['checkpoint'] = str(args.checkpoint)

    # Save final results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {save_path}")

    # Clean up intermediate file
    intermediate_path = f"{save_path}.intermediate"
    if Path(intermediate_path).exists():
        Path(intermediate_path).unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
