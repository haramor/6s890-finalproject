"""
Track Stockfish Agreement Over Training
========================================

Evaluates model checkpoints at different training steps to see how
Stockfish agreement evolves during training.

Tests Stockfish depth 10 (~2400 ELO) at multiple checkpoints.

Usage:
    python stock_agreement_certain_points.py

Output:
    - JSON file with detailed results
    - Terminal output formatted for easy plotting
"""

import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import re

# Add chess-transformers to path
sys.path.insert(0, '/workspace/6s890-finalproject/chess-transformers')

from chess_transformers.transformers.models import ChessTransformerEncoder
from chess_transformers.train.datasets import ChessDataset
from chess_transformers.data.levels import UCI_MOVES

import chess
import chess.engine

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STOCKFISH_PATH = '/usr/games/stockfish'
STOCKFISH_DEPTH = 10

# Reverse mappings
MOVE_INDEX_TO_UCI = {v: k for k, v in UCI_MOVES.items()}
UCI_TO_INDEX = UCI_MOVES.copy()


def extract_step_from_checkpoint(checkpoint_path):
    """Extract step number from checkpoint filename or metadata."""
    # Try to extract from filename first
    filename = Path(checkpoint_path).stem
    
    # Pattern: checkpoint_step_12345.pt or similar
    match = re.search(r'step[_-]?(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Try loading checkpoint metadata
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'step' in checkpoint:
            return checkpoint['step']
    except:
        pass
    
    return None


def load_model_and_config(config_path, checkpoint_path):
    """Load model and config."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    model = ChessTransformerEncoder(config)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'step': checkpoint.get('step', 'unknown'),
        }
    else:
        model.load_state_dict(checkpoint)
        metadata = {'step': extract_step_from_checkpoint(checkpoint_path)}
    
    model.to(DEVICE)
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


def get_stockfish_best_move(board, engine, depth=10):
    """Get Stockfish's best move."""
    try:
        result = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
        if result and len(result) > 0:
            return result[0]['pv'][0].uci()
    except:
        pass
    return None


def evaluate_checkpoint(model, dataloader, n_samples=500):
    """Evaluate Stockfish agreement for one checkpoint."""
    model.eval()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    
    results = {
        'n_evaluated': 0,
        'sf_agreement': 0,
        'legal_moves': 0,
        'top1_correct': 0,
        'top3_correct': 0,
    }
    
    samples_processed = 0
    
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                if samples_processed >= n_samples:
                    break
                
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(DEVICE)
                
                predictions = model(batch)
                pred_moves = predictions[:, 0, :]
                vocab_size = pred_moves.shape[-1]
                actual_moves = batch['moves'][:, 1]
                
                try:
                    boards = batch_to_board_states(batch)
                except:
                    continue
                
                for i, board in enumerate(boards):
                    if samples_processed >= n_samples:
                        break
                    
                    if not board or not board.is_valid():
                        continue
                    
                    legal_moves = list(board.legal_moves)
                    legal_moves_uci = {m.uci() for m in legal_moves}
                    
                    if not legal_moves:
                        continue
                    
                    # Mask illegal moves
                    mask = torch.full((vocab_size,), float('-inf'), device=pred_moves.device)
                    for move in legal_moves:
                        idx = UCI_TO_INDEX.get(move.uci(), None)
                        if idx is not None:
                            mask[idx] = 0.0
                    
                    masked_logits = pred_moves[i] + mask
                    top_indices = torch.topk(masked_logits, min(3, len(legal_moves))).indices.cpu().numpy()
                    
                    model_top1 = MOVE_INDEX_TO_UCI.get(int(top_indices[0]), None)
                    model_top3 = [MOVE_INDEX_TO_UCI.get(int(idx), None) for idx in top_indices]
                    
                    gt_move_uci = MOVE_INDEX_TO_UCI.get(actual_moves[i].item(), None)
                    
                    if not model_top1:
                        continue
                    
                    # Legal move check
                    if model_top1 in legal_moves_uci:
                        results['legal_moves'] += 1
                    
                    # Ground truth accuracy
                    if model_top1 == gt_move_uci:
                        results['top1_correct'] += 1
                    if gt_move_uci in model_top3:
                        results['top3_correct'] += 1
                    
                    # Stockfish agreement
                    sf_best = get_stockfish_best_move(board, engine, depth=STOCKFISH_DEPTH)
                    if sf_best and model_top1 == sf_best:
                        results['sf_agreement'] += 1
                    
                    results['n_evaluated'] += 1
                    samples_processed += 1
    
    finally:
        engine.quit()
    
    # Compute percentages
    n = results['n_evaluated']
    if n > 0:
        return {
            'n_samples': n,
            'legal_move_rate': (results['legal_moves'] / n) * 100,
            'top1_accuracy': (results['top1_correct'] / n) * 100,
            'top3_accuracy': (results['top3_correct'] / n) * 100,
            'sf_agreement_pct': (results['sf_agreement'] / n) * 100,
            'sf_agreement_count': results['sf_agreement'],
        }
    
    return None


def find_checkpoints_in_directory(checkpoint_dir):
    """Find all checkpoints in a directory and sort by step."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []
    
    for ckpt_file in checkpoint_dir.glob('*.pt'):
        step = extract_step_from_checkpoint(ckpt_file)
        if step is not None:
            checkpoints.append((step, ckpt_file))
    
    # Sort by step
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def main():
    print("="*80)
    print("STOCKFISH AGREEMENT OVER TRAINING")
    print(f"Evaluating at Stockfish depth {STOCKFISH_DEPTH} (~2400 ELO)")
    print("="*80)
    
    base_dir = Path('/workspace/6s890-finalproject')
    output_dir = base_dir / 'experiments' / 'scripts' / 'training_progression_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = {
        'baseline': {
            'config': base_dir / 'experiments/configs/baseline_config.py',
            'checkpoint_dir': base_dir / 'experiments/results/baseline_mixed_skill/checkpoints',
            'data_folder': base_dir / 'data',
            'h5_file': 'all_chunks_combined.h5'
        },
        'expert': {
            'config': base_dir / 'experiments/configs/expert_config.py',
            'checkpoint_dir': base_dir / 'experiments/results/expert_LE22ct/checkpoints',
            'data_folder': base_dir / 'data/expert',
            'h5_file': 'LE22ct.h5'
        },
        'random': {
            'config': base_dir / 'experiments/configs/random_config.py',
            'checkpoint_dir': base_dir / 'experiments/results/random_skill/checkpoints',
            'data_folder': base_dir / 'data',
            'h5_file': 'rand_chunks_combined.h5'
        },
    }
    
    all_results = {}
    
    for exp_name, exp_config in experiments.items():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {exp_name.upper()}")
        print(f"{'='*80}")
        
        # Find all checkpoints
        checkpoints = find_checkpoints_in_directory(exp_config['checkpoint_dir'])
        
        if not checkpoints:
            print(f"  ✗ No checkpoints found in {exp_config['checkpoint_dir']}")
            continue
        
        print(f"  Found {len(checkpoints)} checkpoints")
        
        # Select checkpoints to evaluate (evenly spaced + final)
        if len(checkpoints) <= 10:
            selected_checkpoints = checkpoints
        else:
            # Take 10 evenly spaced checkpoints
            indices = np.linspace(0, len(checkpoints)-1, 10, dtype=int)
            selected_checkpoints = [checkpoints[i] for i in indices]
        
        print(f"  Evaluating {len(selected_checkpoints)} checkpoints:")
        for step, ckpt_path in selected_checkpoints:
            print(f"    - Step {step}: {ckpt_path.name}")
        
        # Load dataset once
        dataset = None
        for split_option in ['val', 'test', None]:
            try:
                dataset = ChessDataset(
                    data_folder=str(exp_config['data_folder']),
                    h5_file=exp_config['h5_file'],
                    split=split_option,
                    n_moves=1
                )
                print(f"  ✓ Dataset loaded with split='{split_option}'")
                break
            except:
                continue
        
        if dataset is None:
            print(f"  ✗ Could not load dataset")
            continue
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Evaluate each checkpoint
        exp_results = []
        
        for step, ckpt_path in selected_checkpoints:
            print(f"\n  Evaluating step {step}...")
            
            try:
                model, config, metadata = load_model_and_config(
                    str(exp_config['config']),
                    str(ckpt_path)
                )
            except Exception as e:
                print(f"    ✗ Failed to load: {e}")
                continue
            
            results = evaluate_checkpoint(model, dataloader, n_samples=500)
            
            if results:
                results['step'] = step
                results['checkpoint'] = str(ckpt_path.name)
                exp_results.append(results)
                
                print(f"    Step {step}: SF Agreement = {results['sf_agreement_pct']:.2f}%, "
                      f"Top1 = {results['top1_accuracy']:.2f}%")
        
        all_results[exp_name] = exp_results
        
        # Save individual experiment results
        output_file = output_dir / f'{exp_name}_progression.json'
        with open(output_file, 'w') as f:
            json.dump(exp_results, f, indent=2)
        print(f"\n  ✓ Saved: {output_file}")
    
    # Save combined results
    combined_file = output_dir / 'all_progressions.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved combined results: {combined_file}")
    
    # Print formatted output for plotting
    print(f"\n{'='*80}")
    print("FORMATTED OUTPUT FOR PLOTTING")
    print(f"{'='*80}\n")
    
    for exp_name, exp_results in all_results.items():
        if not exp_results:
            continue
            
        print(f"{exp_name.upper()}:")
        print(f"  Steps: {[r['step'] for r in exp_results]}")
        print(f"  SF Agreement (%): {[round(r['sf_agreement_pct'], 2) for r in exp_results]}")
        print(f"  Top1 Accuracy (%): {[round(r['top1_accuracy'], 2) for r in exp_results]}")
        print(f"  Top3 Accuracy (%): {[round(r['top3_accuracy'], 2) for r in exp_results]}")
        print()
    
    # Print CSV format
    print("\nCSV FORMAT:")
    print("experiment,step,sf_agreement,top1_accuracy,top3_accuracy")
    for exp_name, exp_results in all_results.items():
        for result in exp_results:
            print(f"{exp_name},{result['step']},{result['sf_agreement_pct']:.2f},"
                  f"{result['top1_accuracy']:.2f},{result['top3_accuracy']:.2f}")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
