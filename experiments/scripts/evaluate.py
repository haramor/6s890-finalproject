"""
Comprehensive Evaluation Script for Chess Behavioral Cloning

This script evaluates trained models across multiple metrics:
1. Top-K accuracy (top-1, top-3, top-5)
2. Stockfish alignment (KL-divergence)
3. Centipawn loss
4. Sample complexity analysis

Usage:
    python evaluate.py --checkpoint results/baseline/best_checkpoint.pt --config configs/baseline_config.py
    python evaluate.py --checkpoint results/expert_only/best_checkpoint.pt --config configs/expert_only_config.py
    python evaluate.py --all-conditions  # Evaluate all three conditions
"""

import sys
import os
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import chess
import chess.engine

# Add chess-transformers to path
chess_transformers_path = Path(__file__).parent.parent.parent / "chess-transformers"
sys.path.insert(0, str(chess_transformers_path))

try:
    from chess_transformers.transformers.models import ChessTransformerEncoder
    from chess_transformers.train.datasets import ChessDataset
    from chess_transformers.train.utils import topk_accuracy
except ImportError as e:
    print(f"Warning: Could not import from chess-transformers: {e}")
    print("Some functionality may be limited")


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

    # Handle different checkpoint formats
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


class ChessEvaluator:
    """
    Comprehensive evaluator for chess behavioral cloning models.
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        stockfish_path: Optional[str] = None,
        stockfish_depth: int = 15,
        stockfish_time_limit: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained chess model
            config: Configuration object
            stockfish_path: Path to Stockfish executable (optional)
            stockfish_depth: Search depth for Stockfish
            stockfish_time_limit: Time limit per position (seconds)
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device

        # Initialize Stockfish if path provided
        self.engine = None
        self.stockfish_depth = stockfish_depth
        self.stockfish_time_limit = stockfish_time_limit

        if stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                print(f"Initialized Stockfish from: {stockfish_path}")
            except Exception as e:
                print(f"Warning: Could not initialize Stockfish: {e}")
                print("Stockfish-based metrics will be skipped")

    def evaluate_top_k_accuracy(
        self,
        dataloader: DataLoader,
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Evaluate top-k accuracy on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            k_values: List of k values to compute accuracy for

        Returns:
            Dictionary mapping "top_k" to accuracy values
        """
        print(f"\nEvaluating top-k accuracy (k={k_values})...")

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing predictions"):
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Get predictions
                predictions = self.model(batch)  # (N, n_moves, vocab_size)

                # Extract first move predictions and targets
                pred_first_move = predictions[:, 0, :]  # (N, vocab_size)
                target_first_move = batch["moves"][:, 1]  # (N,)

                all_predictions.append(pred_first_move.cpu())
                all_targets.append(target_first_move.cpu())

        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute top-k accuracies
        results = {}
        for k in k_values:
            top_k_pred = torch.topk(all_predictions, k=k, dim=1)[1]
            correct = (top_k_pred == all_targets.unsqueeze(1)).any(dim=1)
            accuracy = correct.float().mean().item() * 100
            results[f"top_{k}"] = accuracy
            print(f"  Top-{k} accuracy: {accuracy:.2f}%")

        return results

    def evaluate_stockfish_alignment(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = 1000
    ) -> Dict[str, float]:
        """
        Evaluate KL-divergence between model and Stockfish move distributions.

        Args:
            dataloader: DataLoader for evaluation
            max_samples: Maximum number of positions to evaluate (for efficiency)

        Returns:
            Dictionary with mean and std of KL-divergence values
        """
        if self.engine is None:
            print("\nSkipping Stockfish alignment (Stockfish not available)")
            return {"mean_kl": None, "std_kl": None}

        print(f"\nEvaluating Stockfish alignment (max {max_samples} positions)...")

        kl_divergences = []
        samples_processed = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing KL-divergence"):
                if max_samples and samples_processed >= max_samples:
                    break

                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Get predictions
                predictions = self.model(batch)[:, 0, :]  # (N, vocab_size)

                # For each position in batch
                batch_size = predictions.shape[0]
                for i in range(batch_size):
                    if max_samples and samples_processed >= max_samples:
                        break

                    try:
                        # Reconstruct FEN from board position
                        fen = self._reconstruct_fen(batch, i)

                        # Get legal moves
                        board = chess.Board(fen)
                        legal_moves = list(board.legal_moves)

                        if len(legal_moves) == 0:
                            continue

                        # Get Stockfish distribution
                        sf_dist = self._get_stockfish_distribution(board, legal_moves)

                        # Get model distribution
                        model_probs = F.softmax(predictions[i], dim=0)

                        # Compute KL divergence only over legal moves
                        kl = 0.0
                        for move in legal_moves:
                            move_idx = self._move_to_index(move)
                            if move_idx is not None:
                                sf_prob = sf_dist.get(move.uci(), 1e-10)
                                model_prob = model_probs[move_idx].item() + 1e-10
                                kl += sf_prob * np.log(sf_prob / model_prob)

                        kl_divergences.append(kl)
                        samples_processed += 1

                    except Exception as e:
                        # Skip positions that cause errors
                        continue

        if len(kl_divergences) == 0:
            print("  Warning: No valid KL-divergence values computed")
            return {"mean_kl": None, "std_kl": None}

        mean_kl = np.mean(kl_divergences)
        std_kl = np.std(kl_divergences)

        print(f"  Mean KL-divergence: {mean_kl:.4f} ± {std_kl:.4f}")
        print(f"  Positions evaluated: {len(kl_divergences)}")

        return {
            "mean_kl": mean_kl,
            "std_kl": std_kl,
            "n_positions": len(kl_divergences)
        }

    def evaluate_centipawn_loss(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = 1000
    ) -> Dict[str, float]:
        """
        Evaluate centipawn loss (average position evaluation change from predicted moves).

        Args:
            dataloader: DataLoader for evaluation
            max_samples: Maximum number of positions to evaluate

        Returns:
            Dictionary with mean and std of centipawn loss values
        """
        if self.engine is None:
            print("\nSkipping centipawn loss (Stockfish not available)")
            return {"mean_cp_loss": None, "std_cp_loss": None}

        print(f"\nEvaluating centipawn loss (max {max_samples} positions)...")

        cp_losses = []
        samples_processed = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing centipawn loss"):
                if max_samples and samples_processed >= max_samples:
                    break

                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Get predictions
                predictions = self.model(batch)[:, 0, :]  # (N, vocab_size)
                targets = batch["moves"][:, 1]  # (N,)

                # For each position in batch
                batch_size = predictions.shape[0]
                for i in range(batch_size):
                    if max_samples and samples_processed >= max_samples:
                        break

                    try:
                        # Reconstruct position
                        fen = self._reconstruct_fen(batch, i)
                        board = chess.Board(fen)

                        # Get best move according to model
                        pred_move_idx = predictions[i].argmax().item()
                        pred_move = self._index_to_move(pred_move_idx, board)

                        if pred_move is None or pred_move not in board.legal_moves:
                            continue

                        # Evaluate position before move
                        eval_before = self._get_position_eval(board)

                        # Make predicted move
                        board.push(pred_move)
                        eval_after = self._get_position_eval(board)
                        board.pop()

                        # Centipawn loss is the difference in evaluation
                        # (negative if move improves position)
                        cp_loss = eval_before - eval_after
                        cp_losses.append(abs(cp_loss))
                        samples_processed += 1

                    except Exception as e:
                        continue

        if len(cp_losses) == 0:
            print("  Warning: No valid centipawn loss values computed")
            return {"mean_cp_loss": None, "std_cp_loss": None}

        mean_cp = np.mean(cp_losses)
        std_cp = np.std(cp_losses)

        print(f"  Mean centipawn loss: {mean_cp:.2f} ± {std_cp:.2f}")
        print(f"  Positions evaluated: {len(cp_losses)}")

        return {
            "mean_cp_loss": mean_cp,
            "std_cp_loss": std_cp,
            "n_positions": len(cp_losses)
        }

    def _get_stockfish_distribution(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move]
    ) -> Dict[str, float]:
        """Get move distribution from Stockfish."""
        try:
            info = self.engine.analyse(
                board,
                chess.engine.Limit(
                    depth=self.stockfish_depth,
                    time=self.stockfish_time_limit
                ),
                multipv=min(len(legal_moves), 5)
            )

            move_scores = {}
            for result in (info if isinstance(info, list) else [info]):
                if "pv" in result and len(result["pv"]) > 0:
                    move = result["pv"][0].uci()
                    score = result.get("score", chess.engine.Score(0))

                    if score.is_mate():
                        mate_in = score.mate()
                        cp = 10000 * (1 if mate_in > 0 else -1) / abs(mate_in)
                    else:
                        cp = score.score()

                    move_scores[move] = cp

            # Convert to probabilities via softmax
            temperature = 100.0
            moves_uci = [m.uci() for m in legal_moves]
            scores = torch.tensor([move_scores.get(m, -1000) for m in moves_uci])
            probs = F.softmax(scores / temperature, dim=0)

            return {move: probs[i].item() for i, move in enumerate(moves_uci)}

        except Exception as e:
            # Return uniform distribution on error
            uniform_prob = 1.0 / len(legal_moves)
            return {move.uci(): uniform_prob for move in legal_moves}

    def _get_position_eval(self, board: chess.Board) -> float:
        """Get position evaluation in centipawns from Stockfish."""
        try:
            info = self.engine.analyse(
                board,
                chess.engine.Limit(depth=self.stockfish_depth)
            )
            score = info.get("score", chess.engine.Score(0))

            if score.is_mate():
                mate_in = score.mate()
                return 10000 * (1 if mate_in > 0 else -1)
            else:
                return score.score()
        except:
            return 0.0

    def _reconstruct_fen(self, batch: Dict, idx: int) -> str:
        """
        Reconstruct FEN string from batch data.
        This is a simplified version - you may need to adjust based on your data format.
        """
        # This is a placeholder - you'll need to implement based on your data format
        # For now, return starting position
        return chess.STARTING_FEN

    def _move_to_index(self, move: chess.Move) -> Optional[int]:
        """Convert chess.Move to vocabulary index."""
        # This is a placeholder - you'll need the actual vocabulary mapping
        # from chess-transformers
        return None

    def _index_to_move(self, idx: int, board: chess.Board) -> Optional[chess.Move]:
        """Convert vocabulary index to chess.Move."""
        # This is a placeholder - you'll need the actual vocabulary mapping
        return None

    def evaluate_all(
        self,
        test_loader: DataLoader,
        max_stockfish_samples: int = 1000
    ) -> Dict:
        """
        Run all evaluation metrics.

        Args:
            test_loader: DataLoader for test set
            max_stockfish_samples: Max samples for Stockfish-based metrics

        Returns:
            Dictionary with all evaluation results
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION")
        print("="*70)

        results = {}

        # Top-K accuracy
        results.update(self.evaluate_top_k_accuracy(test_loader))

        # Stockfish alignment
        results.update(self.evaluate_stockfish_alignment(
            test_loader,
            max_samples=max_stockfish_samples
        ))

        # Centipawn loss
        results.update(self.evaluate_centipawn_loss(
            test_loader,
            max_samples=max_stockfish_samples
        ))

        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)

        return results

    def __del__(self):
        """Clean up Stockfish engine."""
        if self.engine is not None:
            try:
                self.engine.quit()
            except:
                pass


def evaluate_single_condition(
    config_path: str,
    checkpoint_path: str,
    stockfish_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate a single experimental condition.

    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint
        stockfish_path: Path to Stockfish executable (optional)
        output_dir: Directory to save results (optional)

    Returns:
        Dictionary with evaluation results
    """
    # Load config
    config = load_config(config_path)

    print(f"\n{'='*70}")
    print(f"Evaluating: {config.NAME}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}")

    # Create test dataloader
    try:
        test_loader = DataLoader(
            ChessDataset(
                data_folder=config.DATA_FOLDER,
                h5_file=config.H5_FILE,
                split="val",  # Using validation set as test set for now
                n_moves=config.N_MOVES
            ),
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
            pin_memory=False
        )
        print(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return {}

    # Load model
    try:
        model = ChessTransformerEncoder(config)
    except:
        # Use simplified model if ChessTransformerEncoder not available
        print("Warning: Using simplified test model")
        model = nn.Module()  # Placeholder

    # Load checkpoint
    load_checkpoint(checkpoint_path, model)

    # Create evaluator
    evaluator = ChessEvaluator(
        model=model,
        config=config,
        stockfish_path=stockfish_path,
        device=DEVICE
    )

    # Run evaluation
    results = evaluator.evaluate_all(test_loader)

    # Add metadata
    results['config_name'] = config.NAME
    results['experiment_type'] = config.EXPERIMENT_TYPE
    results['checkpoint'] = checkpoint_path

    # Save results if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / f"{config.NAME}_evaluation.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate chess behavioral cloning models"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=os.environ.get("CT_STOCKFISH_PATH", "/usr/local/bin/stockfish"),
        help="Path to Stockfish executable"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="Evaluate all three experimental conditions"
    )

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    if args.all_conditions:
        # Evaluate all three conditions
        conditions = [
            {
                'config': base_dir / 'configs' / 'baseline_config.py',
                'checkpoint': base_dir / 'results' / 'baseline_mixed_skill' / 'checkpoints' / 'best_checkpoint.pt'
            },
            {
                'config': base_dir / 'configs' / 'expert_only_config.py',
                'checkpoint': base_dir / 'results' / 'expert_only_2500' / 'checkpoints' / 'best_checkpoint.pt'
            },
            {
                'config': base_dir / 'configs' / 'game_theoretic_config.py',
                'checkpoint': base_dir / 'results' / 'game_theoretic_reg' / 'checkpoints' / 'best_checkpoint.pt'
            }
        ]

        all_results = {}
        for condition in conditions:
            if condition['checkpoint'].exists():
                results = evaluate_single_condition(
                    config_path=str(condition['config']),
                    checkpoint_path=str(condition['checkpoint']),
                    stockfish_path=args.stockfish_path,
                    output_dir=args.output_dir
                )
                all_results[results['config_name']] = results
            else:
                print(f"\nWarning: Checkpoint not found: {condition['checkpoint']}")

        # Print summary comparison
        print("\n" + "="*70)
        print("SUMMARY COMPARISON")
        print("="*70)
        for name, results in all_results.items():
            print(f"\n{name}:")
            print(f"  Top-1 accuracy: {results.get('top_1', 'N/A')}")
            print(f"  Top-3 accuracy: {results.get('top_3', 'N/A')}")
            print(f"  KL-divergence: {results.get('mean_kl', 'N/A')}")
            print(f"  Centipawn loss: {results.get('mean_cp_loss', 'N/A')}")

    else:
        # Evaluate single condition
        if not args.config or not args.checkpoint:
            parser.error("--config and --checkpoint required (or use --all-conditions)")

        evaluate_single_condition(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            stockfish_path=args.stockfish_path,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
