"""
Advanced Chess Model Evaluation
Computes KL-divergence, centipawn loss, and detailed Stockfish alignment metrics
"""

import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

# Add chess-transformers to path
sys.path.insert(0, '/workspace/6s890-finalproject/chess-transformers')

from chess_transformers.transformers.models import ChessTransformerEncoder
from chess_transformers.train.datasets import ChessDataset
from chess_transformers.data.levels import UCI_MOVES, PIECES, TURN

import chess
import chess.engine

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STOCKFISH_PATH = '/usr/games/stockfish'

# Reverse mappings
MOVE_INDEX_TO_UCI = {v: k for k, v in UCI_MOVES.items()}
UCI_TO_INDEX = UCI_MOVES.copy()


def load_model_and_config(config_path, checkpoint_path):
    """Load model and config."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    model = ChessTransformerEncoder(config)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'step': checkpoint.get('step', 'unknown'),
        }
    else:
        model.load_state_dict(checkpoint)
        metadata = {}
    
    model.eval()
    return model, config, metadata


def decode_board_position(encoded_board):
    """Convert encoded board position to chess.Board."""
    board = chess.Board()
    board.clear()
    
    piece_map = {
        2: chess.Piece(chess.PAWN, chess.WHITE),
        3: chess.Piece(chess.PAWN, chess.BLACK),
        4: chess.Piece(chess.ROOK, chess.WHITE),
        5: chess.Piece(chess.ROOK, chess.BLACK),
        6: chess.Piece(chess.KNIGHT, chess.WHITE),
        7: chess.Piece(chess.KNIGHT, chess.BLACK),
        8: chess.Piece(chess.BISHOP, chess.WHITE),
        9: chess.Piece(chess.BISHOP, chess.BLACK),
        10: chess.Piece(chess.QUEEN, chess.WHITE),
        11: chess.Piece(chess.QUEEN, chess.BLACK),
        12: chess.Piece(chess.KING, chess.WHITE),
        13: chess.Piece(chess.KING, chess.BLACK),
    }
    
    for dataset_idx in range(64):
        piece_code = int(encoded_board[dataset_idx])
        if piece_code in piece_map:
            dataset_file = dataset_idx % 8
            dataset_rank = dataset_idx // 8
            chess_rank = 7 - dataset_rank
            chess_idx = chess_rank * 8 + dataset_file
            board.set_piece_at(chess_idx, piece_map[piece_code])
    
    return board


def batch_to_board_states(batch):
    """Convert batch data to list of chess boards."""
    boards = []
    batch_size = batch['board_positions'].shape[0]
    
    for i in range(batch_size):
        board_encoded = batch['board_positions'][i].cpu().numpy()
        board = decode_board_position(board_encoded)
        
        turn_code = batch['turns'][i].item()
        board.turn = chess.WHITE if turn_code == 1 else chess.BLACK
        
        board.castling_rights = 0
        if batch['white_kingside_castling_rights'][i].item() == 1:
            board.castling_rights |= chess.BB_H1
        if batch['white_queenside_castling_rights'][i].item() == 1:
            board.castling_rights |= chess.BB_A1
        if batch['black_kingside_castling_rights'][i].item() == 1:
            board.castling_rights |= chess.BB_H8
        if batch['black_queenside_castling_rights'][i].item() == 1:
            board.castling_rights |= chess.BB_A8
        
        boards.append(board)
    
    return boards


def get_stockfish_distribution(board, engine, depth=15, multipv=20):
    """
    Get Stockfish's move distribution for a position.
    
    Returns:
        dict: {move_uci: probability} for top moves
        int: centipawn evaluation of best move
    """
    try:
        # Get top moves with evaluations
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        
        if not info:
            return None, None
        
        # Extract scores (in centipawns)
        moves_scores = []
        for result in info:
            move = result['pv'][0]
            score = result['score'].relative
            
            # Convert score to centipawns
            if score.is_mate():
                # Mate scores: use very large values
                cp = 10000 if score.mate() > 0 else -10000
            else:
                cp = score.score()
            
            moves_scores.append((move.uci(), cp))
        
        if not moves_scores:
            return None, None
        
        best_cp = moves_scores[0][1]
        
        # Convert centipawn scores to probabilities using softmax with temperature
        # Higher temperature = more uniform distribution
        temperature = 100.0  # Standard for chess
        
        scores = np.array([score for _, score in moves_scores])
        scores = scores / temperature
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        probabilities = exp_scores / exp_scores.sum()
        
        # Create distribution dict
        distribution = {move: float(prob) for (move, _), prob in zip(moves_scores, probabilities)}
        
        return distribution, best_cp
        
    except Exception as e:
        print(f"Warning: Stockfish analysis failed: {e}")
        return None, None


def compute_kl_divergence(model_dist, stockfish_dist, legal_moves_uci):
    """
    Compute KL divergence: KL(model || stockfish)
    
    Args:
        model_dist: dict {move_uci: probability}
        stockfish_dist: dict {move_uci: probability}
        legal_moves_uci: set of legal move UCI strings
    
    Returns:
        float: KL divergence value
    """
    # Get all moves that appear in either distribution
    all_moves = set(model_dist.keys()) | set(stockfish_dist.keys())
    all_moves = all_moves & legal_moves_uci  # Only legal moves
    
    if not all_moves:
        return None
    
    kl_div = 0.0
    epsilon = 1e-10  # Small constant to avoid log(0)
    
    for move in all_moves:
        p_model = model_dist.get(move, epsilon)
        p_sf = stockfish_dist.get(move, epsilon)
        
        if p_model > epsilon:
            kl_div += p_model * np.log(p_model / p_sf)
    
    return kl_div


def get_model_distribution(logits, board, vocab_size, top_k=20):
    """
    Get model's move distribution for a position.
    
    Args:
        logits: (vocab_size,) tensor of model logits
        board: chess.Board
        vocab_size: int
        top_k: number of top moves to consider
    
    Returns:
        dict: {move_uci: probability} for legal moves
    """
    # Mask illegal moves
    mask = torch.full((vocab_size,), float('-inf'), device=logits.device)
    
    for move in board.legal_moves:
        uci = move.uci()
        idx = UCI_TO_INDEX.get(uci, None)
        if idx is not None and 0 <= idx < vocab_size:
            mask[idx] = 0.0
    
    # Apply mask and get probabilities
    masked_logits = logits + mask
    probs = F.softmax(masked_logits, dim=0).cpu().numpy()
    
    # Get top-k moves
    top_indices = np.argsort(probs)[-top_k:][::-1]
    
    distribution = {}
    for idx in top_indices:
        move_uci = MOVE_INDEX_TO_UCI.get(int(idx), None)
        if move_uci and probs[idx] > 1e-10:
            distribution[move_uci] = float(probs[idx])
    
    # Renormalize to sum to 1
    total = sum(distribution.values())
    if total > 0:
        distribution = {k: v/total for k, v in distribution.items()}
    
    return distribution


def compute_centipawn_loss(board, model_move, engine, depth=15):
    """
    Compute centipawn loss for a move.
    
    Centipawn loss = evaluation before move - evaluation after move
    
    Returns:
        int: centipawn loss (0 = perfect move, higher = worse)
    """
    try:
        # Evaluate current position
        info_before = engine.analyse(board, chess.engine.Limit(depth=depth))
        score_before = info_before['score'].relative
        
        if score_before.is_mate():
            cp_before = 10000 if score_before.mate() > 0 else -10000
        else:
            cp_before = score_before.score()
        
        # Make the model's move
        board_copy = board.copy()
        board_copy.push(model_move)
        
        # Evaluate after move (from opponent's perspective, so negate)
        info_after = engine.analyse(board_copy, chess.engine.Limit(depth=depth))
        score_after = info_after['score'].relative
        
        if score_after.is_mate():
            cp_after = 10000 if score_after.mate() > 0 else -10000
        else:
            cp_after = score_after.score()
        
        # Centipawn loss (negative because we negate opponent's eval)
        cp_loss = cp_before - (-cp_after)
        
        return max(0, cp_loss)  # Loss is always non-negative
        
    except Exception as e:
        print(f"Warning: Centipawn loss computation failed: {e}")
        return None


def advanced_evaluation(model, dataloader, n_samples=500, stockfish_depth=15):
    """
    Comprehensive evaluation including KL-divergence and centipawn loss.
    """
    print(f"\n{'='*70}")
    print(f"Advanced Evaluation: {n_samples} positions, Stockfish depth {stockfish_depth}")
    print(f"{'='*70}")
    
    model.to(DEVICE)
    model.eval()
    
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    
    results = {
        'n_evaluated': 0,
        'kl_divergences': [],
        'centipawn_losses': [],
        'stockfish_agreement_exact': 0,
        'model_in_sf_top3': 0,
        'model_in_sf_top5': 0,
        'sf_in_model_top3': 0,
        'sf_in_model_top5': 0,
        'top1_accuracy': 0,
    }
    
    samples_processed = 0
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if samples_processed >= n_samples:
                    break
                
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(DEVICE)
                
                batch_size = batch['moves'].shape[0]
                
                # Model predictions
                predictions = model(batch)
                pred_moves = predictions[:, 0, :]
                vocab_size = pred_moves.shape[-1]
                
                # Ground truth
                actual_move = batch['moves'][:, 1]
                
                # Decode boards
                try:
                    boards = batch_to_board_states(batch)
                except Exception as e:
                    print(f"\nWarning: Board decoding failed: {e}")
                    samples_processed += batch_size
                    continue
                
                # Evaluate each position
                for i in range(min(batch_size, len(boards))):
                    if samples_processed >= n_samples:
                        break
                    
                    board = boards[i]
                    if not board or not board.is_valid():
                        samples_processed += 1
                        continue
                    
                    # Get legal moves
                    legal_moves = list(board.legal_moves)
                    legal_moves_uci = {m.uci() for m in legal_moves}
                    
                    if not legal_moves:
                        samples_processed += 1
                        continue
                    
                    # Get model distribution
                    model_dist = get_model_distribution(
                        pred_moves[i], board, vocab_size, top_k=20
                    )
                    
                    # Get model's top-1 move
                    mask = torch.full((vocab_size,), float('-inf'), device=pred_moves.device)
                    for move in legal_moves:
                        idx = UCI_TO_INDEX.get(move.uci(), None)
                        if idx is not None:
                            mask[idx] = 0.0
                    
                    masked_logits = pred_moves[i] + mask
                    top1_idx = int(torch.argmax(masked_logits).item())
                    model_move_uci = MOVE_INDEX_TO_UCI.get(top1_idx, None)
                    
                    if not model_move_uci:
                        samples_processed += 1
                        continue
                    
                    try:
                        model_move = chess.Move.from_uci(model_move_uci)
                    except:
                        samples_processed += 1
                        continue
                    
                    if model_move not in legal_moves:
                        samples_processed += 1
                        continue
                    
                    # Get Stockfish distribution
                    sf_dist, best_cp = get_stockfish_distribution(
                        board, engine, depth=stockfish_depth, multipv=20
                    )
                    
                    if sf_dist is None:
                        samples_processed += 1
                        continue
                    
                    # Compute KL divergence
                    kl_div = compute_kl_divergence(model_dist, sf_dist, legal_moves_uci)
                    if kl_div is not None:
                        results['kl_divergences'].append(kl_div)
                    
                    # Compute centipawn loss
                    cp_loss = compute_centipawn_loss(
                        board, model_move, engine, depth=stockfish_depth
                    )
                    if cp_loss is not None:
                        results['centipawn_losses'].append(cp_loss)
                    
                    # Stockfish agreement metrics
                    sf_best_move_uci = list(sf_dist.keys())[0] if sf_dist else None
                    sf_top5_moves = list(sf_dist.keys())[:5]
                    sf_top3_moves = list(sf_dist.keys())[:3]
                    
                    if model_move_uci == sf_best_move_uci:
                        results['stockfish_agreement_exact'] += 1
                    
                    if model_move_uci in sf_top3_moves:
                        results['model_in_sf_top3'] += 1
                    
                    if model_move_uci in sf_top5_moves:
                        results['model_in_sf_top5'] += 1
                    
                    # Model top-k
                    model_top5 = list(model_dist.keys())[:5]
                    model_top3 = list(model_dist.keys())[:3]
                    
                    if sf_best_move_uci in model_top3:
                        results['sf_in_model_top3'] += 1
                    
                    if sf_best_move_uci in model_top5:
                        results['sf_in_model_top5'] += 1
                    
                    # Top-1 accuracy
                    if top1_idx == actual_move[i].item():
                        results['top1_accuracy'] += 1
                    
                    results['n_evaluated'] += 1
                    samples_processed += 1
                
    finally:
        engine.quit()
    
    # Compute statistics
    n = results['n_evaluated']
    if n > 0:
        results['mean_kl_divergence'] = float(np.mean(results['kl_divergences'])) if results['kl_divergences'] else 0.0
        results['std_kl_divergence'] = float(np.std(results['kl_divergences'])) if results['kl_divergences'] else 0.0
        results['median_kl_divergence'] = float(np.median(results['kl_divergences'])) if results['kl_divergences'] else 0.0
        
        results['mean_centipawn_loss'] = float(np.mean(results['centipawn_losses'])) if results['centipawn_losses'] else 0.0
        results['std_centipawn_loss'] = float(np.std(results['centipawn_losses'])) if results['centipawn_losses'] else 0.0
        results['median_centipawn_loss'] = float(np.median(results['centipawn_losses'])) if results['centipawn_losses'] else 0.0
        
        results['stockfish_agreement_exact'] = (results['stockfish_agreement_exact'] / n) * 100
        results['model_in_sf_top3'] = (results['model_in_sf_top3'] / n) * 100
        results['model_in_sf_top5'] = (results['model_in_sf_top5'] / n) * 100
        results['sf_in_model_top3'] = (results['sf_in_model_top3'] / n) * 100
        results['sf_in_model_top5'] = (results['sf_in_model_top5'] / n) * 100
        results['top1_accuracy'] = (results['top1_accuracy'] / n) * 100
    
    return results


def main():
    print("="*70)
    print("ADVANCED CHESS EVALUATION")
    print("="*70)
    
    output_dir = Path('/workspace/6s890-finalproject/results/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not Path(STOCKFISH_PATH).exists():
        print(f"ERROR: Stockfish not found at {STOCKFISH_PATH}")
        return
    
    experiments = {
        'baseline': {
            'config': '/workspace/6s890-finalproject/experiments/configs/baseline_config.py',
            'checkpoint': '/workspace/6s890-finalproject/experiments/results/baseline_mixed_skill/checkpoints/best_baseline_mixed_skill.pt',
            'data_folder': '/workspace/6s890-finalproject/data',
            'h5_file': 'all_chunks_combined.h5'
        },
        'expert': {
            'config': '/workspace/6s890-finalproject/experiments/configs/expert_config.py',
            'checkpoint': '/workspace/6s890-finalproject/experiments/results/expert_LE22ct/checkpoints/best_expert_LE22ct.pt',
            'data_folder': '/workspace/6s890-finalproject/data/expert',
            'h5_file': 'LE22ct.h5'
        }
    }
    
    all_results = {}
    
    for exp_name, exp_config in experiments.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {exp_name.upper()}")
        print(f"{'='*70}")
        
        model, config, metadata = load_model_and_config(
            exp_config['config'], 
            exp_config['checkpoint']
        )
        print(f"✓ Model loaded (step: {metadata.get('step', 'unknown')})")
        
        dataset = ChessDataset(
            data_folder=exp_config['data_folder'],
            h5_file=exp_config['h5_file'],
            split='val',
            n_moves=1
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=8,  # Smaller batch for Stockfish multipv
            shuffle=False,
            num_workers=0
        )
        
        print(f"✓ Dataset loaded ({len(dataset)} positions)")
        
        results = advanced_evaluation(
            model,
            dataloader,
            n_samples=500,  # Fewer samples since this is computationally expensive
            stockfish_depth=15
        )
        
        results['metadata'] = metadata
        results['experiment'] = exp_name
        
        all_results[exp_name] = results
        
        # Save individual results
        output_file = output_dir / f'{exp_name}_advanced_eval.json'
        with open(output_file, 'w') as f:
            # Remove raw lists before saving
            save_results = results.copy()
            save_results.pop('kl_divergences', None)
            save_results.pop('centipawn_losses', None)
            json.dump(save_results, f, indent=2)
        print(f"✓ Saved: {output_file}")
    
    # Save comparison
    comparison_file = output_dir / 'advanced_eval_comparison.json'
    comparison_data = {}
    for exp_name, results in all_results.items():
        save_results = results.copy()
        save_results.pop('kl_divergences', None)
        save_results.pop('centipawn_losses', None)
        comparison_data[exp_name] = save_results
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"✓ Saved comparison: {comparison_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    
    for exp_name, results in all_results.items():
        print(f"\n{exp_name.upper()}:")
        print(f"  Positions Evaluated: {results['n_evaluated']}")
        print(f"  Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
        print(f"  Stockfish Agreement (exact): {results['stockfish_agreement_exact']:.2f}%")
        print(f"  Model in SF Top-5: {results['model_in_sf_top5']:.2f}%")
        print(f"  SF in Model Top-5: {results['sf_in_model_top5']:.2f}%")
        print(f"\n  KL-Divergence (model || stockfish):")
        print(f"    Mean: {results['mean_kl_divergence']:.4f}")
        print(f"    Median: {results['median_kl_divergence']:.4f}")
        print(f"    Std: {results['std_kl_divergence']:.4f}")
        print(f"\n  Centipawn Loss:")
        print(f"    Mean: {results['mean_centipawn_loss']:.1f} cp")
        print(f"    Median: {results['median_centipawn_loss']:.1f} cp")
        print(f"    Std: {results['std_centipawn_loss']:.1f} cp")
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
