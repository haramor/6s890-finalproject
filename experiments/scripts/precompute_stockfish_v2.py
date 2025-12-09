"""
Precompute Stockfish Evaluations for Game-Theoretic Training

This script precomputes Stockfish move distributions for all positions in the
training dataset and saves them to disk. This allows game-theoretic training
to use Stockfish evaluations without the overhead of calling Stockfish during
training.

Usage:
    python precompute_stockfish.py --data_dir data/expert_2500 --h5_file expert_2500_10k.h5 --output stockfish_cache.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np

import chess
import chess.engine
import torch
import torch.nn.functional as F

# Add chess-transformers to path
chess_transformers_path = Path(__file__).parent.parent.parent / "chess-transformers"
sys.path.insert(0, str(chess_transformers_path))

try:
    from chess_transformers.train.datasets import ChessDataset
except ImportError:
    print("Warning: Could not import ChessDataset")


def board_tensor_to_fen(board_tensor):
    """
    Convert board position tensor to FEN string.
    
    Args:
        board_tensor: Tensor of shape (64,) with piece encodings
                     Encoding: 0=empty, 1=P, 2=N, 3=B, 4=R, 5=Q, 6=K,
                              7=p, 8=n, 9=b, 10=r, 11=q, 12=k
    
    Returns:
        FEN string representing the position
    """
    piece_map = {
        0: '.',   # empty
        1: 'P',   # white pawn
        2: 'N',   # white knight
        3: 'B',   # white bishop
        4: 'R',   # white rook
        5: 'Q',   # white queen
        6: 'K',   # white king
        7: 'p',   # black pawn
        8: 'n',   # black knight
        9: 'b',   # black bishop
        10: 'r',  # black rook
        11: 'q',  # black queen
        12: 'k',  # black king
    }

    # Convert to numpy if needed
    if isinstance(board_tensor, torch.Tensor):
        board_array = board_tensor.cpu().numpy()
    else:
        board_array = board_tensor

    # Build FEN string rank by rank (from rank 8 to rank 1)
    fen_ranks = []
    for rank in range(7, -1, -1):  # Ranks 8 to 1
        fen_rank = ""
        empty_count = 0
        
        for file in range(8):  # Files a to h
            square_idx = rank * 8 + file
            piece_code = int(board_array[square_idx])
            piece_char = piece_map.get(piece_code, '.')
            
            if piece_char == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_rank += str(empty_count)
                    empty_count = 0
                fen_rank += piece_char
        
        if empty_count > 0:
            fen_rank += str(empty_count)
        
        fen_ranks.append(fen_rank)

    # Join ranks with '/'
    board_fen = '/'.join(fen_ranks)

    # For simplicity, assume white to move, no castling rights, no en passant
    full_fen = f"{board_fen} w KQkq - 0 1"

    return full_fen


def get_stockfish_distribution(engine, fen, depth=15, time_limit=0.1, temperature=100.0):
    """
    Get move distribution from Stockfish for a given position.
    
    Args:
        engine: Stockfish engine instance
        fen: Position in FEN notation
        depth: Search depth
        time_limit: Time limit per position (seconds)
        temperature: Softmax temperature for converting scores to probabilities
    
    Returns:
        Dictionary mapping UCI moves to probabilities
    """
    try:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        
        if len(legal_moves) == 0:
            return {}

        # Analyze with multi-PV to get top moves
        info = engine.analyse(
            board,
            chess.engine.Limit(depth=depth, time=time_limit),
            multipv=min(len(legal_moves), 5)  # Top 5 moves
        )

        # Extract centipawn scores and moves
        move_scores = {}
        for result in (info if isinstance(info, list) else [info]):
            if "pv" in result and len(result["pv"]) > 0:
                move = result["pv"][0].uci()
                score = result.get("score", chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))

                # Convert score to centipawns
                pov_score = score.white() if board.turn == chess.WHITE else score.black()
                
                if pov_score.is_mate():
                    mate_in = pov_score.mate()
                    cp = 10000 if mate_in > 0 else -10000
                else:
                    cp = pov_score.score()

                move_scores[move] = cp

        # If no moves were evaluated, return uniform
        if not move_scores:
            uniform_prob = 1.0 / len(legal_moves)
            return {move.uci(): uniform_prob for move in legal_moves}

        # Convert centipawn scores to probabilities via softmax
        legal_moves_uci = [move.uci() for move in legal_moves]
        default_score = min(move_scores.values()) - 500 if move_scores else -1000
        scores = torch.tensor([move_scores.get(m, default_score) for m in legal_moves_uci])
        probs = F.softmax(scores / temperature, dim=0)

        return {move: probs[i].item() for i, move in enumerate(legal_moves_uci)}

    except Exception as e:
        print(f"Error evaluating position: {e}")
        return {}


def precompute_stockfish_evaluations(
    data_dir,
    h5_file,
    output_file,
    stockfish_path="/usr/games/stockfish",
    depth=15,
    time_limit=0.1,
    temperature=100.0
):
    """
    Precompute Stockfish evaluations for all positions in the dataset.
    
    Args:
        data_dir: Directory containing the h5 file
        h5_file: Name of the h5 file
        output_file: Path to save the precomputed distributions
        stockfish_path: Path to Stockfish executable
        depth: Search depth
        time_limit: Time limit per position
        temperature: Softmax temperature
    """
    print("="*60)
    print("Precomputing Stockfish Evaluations")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"H5 file: {h5_file}")
    print(f"Output file: {output_file}")
    print(f"Stockfish path: {stockfish_path}")
    print(f"Depth: {depth}")
    print(f"Time limit: {time_limit}s per position")
    print("="*60)
    
    # Load dataset
    print(f"\nLoading dataset from {data_dir}/{h5_file}...")
    try:
        dataset = ChessDataset(
            data_folder=data_dir,
            h5_file=h5_file,
            split="train",
            n_moves=1
        )
        print(f"✓ Loaded {len(dataset)} training positions")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # Precompute distributions
    print(f"\nPrecomputing Stockfish distributions...")
    distributions = {}
    errors = 0
    engine = None
    
    # Restart engine every N positions to prevent crashes
    RESTART_INTERVAL = 50
    
    try:
        for i in tqdm(range(len(dataset)), desc="Evaluating positions"):
            try:
                # Restart engine periodically
                if i % RESTART_INTERVAL == 0:
                    if engine is not None:
                        try:
                            engine.quit()
                        except:
                            pass
                    
                    # Start new engine
                    try:
                        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                    except Exception as e:
                        print(f"\nFailed to start Stockfish: {e}")
                        errors += 1
                        continue
                
                # Get board position
                sample = dataset[i]
                board_positions = sample['board_positions']
                
                # Convert to FEN
                fen = board_tensor_to_fen(board_positions)
                
                # Skip if already computed
                if fen in distributions:
                    continue
                
                # Get Stockfish distribution
                if engine is not None:
                    dist = get_stockfish_distribution(
                        engine, fen, depth, time_limit, temperature
                    )
                    
                    if dist:
                        distributions[fen] = dist
                    else:
                        errors += 1
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"\nError processing position {i}: {e}")
    
    finally:
        # Clean up engine
        if engine is not None:
            try:
                engine.quit()
            except:
                pass
    
    # Save distributions
    print(f"\nSaving distributions to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(distributions, f)
    
    print("="*60)
    print("Precomputation Complete!")
    print(f"✓ Evaluated {len(distributions)} unique positions")
    print(f"✗ Errors: {errors}")
    print(f"✓ Saved to: {output_file}")
    print(f"Cache size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Precompute Stockfish evaluations")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/expert_2500",
        help="Directory containing the h5 file"
    )
    parser.add_argument(
        "--h5_file",
        type=str,
        default="expert_2500_10k.h5",
        help="Name of the h5 file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stockfish_cache.pkl",
        help="Output file for precomputed distributions"
    )
    parser.add_argument(
        "--stockfish_path",
        type=str,
        default="/usr/games/stockfish",
        help="Path to Stockfish executable"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=15,
        help="Stockfish search depth"
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=0.1,
        help="Time limit per position (seconds)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=100.0,
        help="Softmax temperature for probability distribution"
    )
    
    args = parser.parse_args()
    
    precompute_stockfish_evaluations(
        data_dir=args.data_dir,
        h5_file=args.h5_file,
        output_file=args.output,
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        time_limit=args.time_limit,
        temperature=args.temperature
    )


if __name__ == "__main__":
    main()
