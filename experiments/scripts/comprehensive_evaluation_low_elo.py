"""
Comprehensive Chess Model Evaluation - Lower ELO Stockfish
Run from: /workspace/6s890-finalproject/experiments/scripts/

Uses lower Stockfish depths (club to master level):
- All models tested at depths [5, 8, 10]
- Depth 5: ~2000 ELO (club player)
- Depth 8: ~2200 ELO (strong club/expert)
- Depth 10: ~2400 ELO (master)
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
from chess_transformers.data.levels import UCI_MOVES

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
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'step': checkpoint.get('step', 'unknown'),
        }
    else:
        model.load_state_dict(checkpoint)
        metadata = {}
    
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


def get_stockfish_top_moves(board, engine, depth=15, num_moves=5):
    """Get Stockfish's top moves for a position."""
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=num_moves)
        if not info:
            return []
        
        top_moves = []
        for result in info:
            move = result['pv'][0]
            top_moves.append(move.uci())
        
        return top_moves
    except Exception as e:
        return []


def comprehensive_evaluation(model, dataloader, n_samples=1000, stockfish_depths=[5, 8, 10]):
    """Comprehensive evaluation with all requested metrics."""
    print(f"\n{'='*70}")
    print(f"Evaluating {n_samples} positions")
    print(f"Stockfish depths: {stockfish_depths}")
    
    # Show ELO estimates
    depth_to_elo = {5: "~2000", 8: "~2200", 10: "~2400", 12: "~2600", 15: "~2700-2800"}
    for depth in stockfish_depths:
        elo_est = depth_to_elo.get(depth, "~2500+")
        print(f"  Depth {depth}: {elo_est} ELO")
    
    print(f"{'='*70}\n")
    
    model.eval()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    
    results = {
        'n_evaluated': 0,
        'legal_moves_total': 0,
        'legal_moves_correct': 0,
        'top1_correct': 0,
        'top3_correct': 0,
        'top5_correct': 0,
        'stockfish_depths_used': stockfish_depths,
    }
    
    for depth in stockfish_depths:
        results[f'sf_agreement_depth{depth}'] = 0
        results[f'model_in_sf_top3_depth{depth}'] = 0
        results[f'model_in_sf_top5_depth{depth}'] = 0
        results[f'sf_in_model_top3_depth{depth}'] = 0
        results[f'sf_in_model_top5_depth{depth}'] = 0
    
    samples_processed = 0
    
    try:
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluating")
            for batch in pbar:
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
                except Exception as e:
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
                    top_indices = torch.topk(masked_logits, min(5, len(legal_moves))).indices.cpu().numpy()
                    
                    model_top1 = MOVE_INDEX_TO_UCI.get(int(top_indices[0]), None)
                    model_top3 = [MOVE_INDEX_TO_UCI.get(int(idx), None) for idx in top_indices[:min(3, len(top_indices))]]
                    model_top5 = [MOVE_INDEX_TO_UCI.get(int(idx), None) for idx in top_indices[:min(5, len(top_indices))]]
                    
                    gt_move_uci = MOVE_INDEX_TO_UCI.get(actual_moves[i].item(), None)
                    
                    if not model_top1:
                        continue
                    
                    # Legal move rate
                    results['legal_moves_total'] += 1
                    if model_top1 in legal_moves_uci:
                        results['legal_moves_correct'] += 1
                    
                    # Accuracy vs ground truth
                    if model_top1 == gt_move_uci:
                        results['top1_correct'] += 1
                    if gt_move_uci in model_top3:
                        results['top3_correct'] += 1
                    if gt_move_uci in model_top5:
                        results['top5_correct'] += 1
                    
                    # Stockfish alignment
                    for depth in stockfish_depths:
                        sf_top = get_stockfish_top_moves(board, engine, depth=depth, num_moves=5)
                        if not sf_top:
                            continue
                        
                        sf_best = sf_top[0]
                        sf_top3 = sf_top[:3]
                        sf_top5 = sf_top
                        
                        if model_top1 == sf_best:
                            results[f'sf_agreement_depth{depth}'] += 1
                        if model_top1 in sf_top3:
                            results[f'model_in_sf_top3_depth{depth}'] += 1
                        if model_top1 in sf_top5:
                            results[f'model_in_sf_top5_depth{depth}'] += 1
                        if sf_best in model_top3:
                            results[f'sf_in_model_top3_depth{depth}'] += 1
                        if sf_best in model_top5:
                            results[f'sf_in_model_top5_depth{depth}'] += 1
                    
                    results['n_evaluated'] += 1
                    samples_processed += 1
                    
                    pbar.set_postfix({
                        'n': samples_processed,
                        'legal%': 100 * results['legal_moves_correct'] / max(1, results['legal_moves_total']),
                        'top1%': 100 * results['top1_correct'] / max(1, results['n_evaluated'])
                    })
    finally:
        engine.quit()
    
    # Compute percentages
    n = results['n_evaluated']
    if n > 0:
        results['legal_move_rate'] = (results['legal_moves_correct'] / results['legal_moves_total']) * 100
        results['top1_accuracy'] = (results['top1_correct'] / n) * 100
        results['top3_accuracy'] = (results['top3_correct'] / n) * 100
        results['top5_accuracy'] = (results['top5_correct'] / n) * 100
        
        for depth in stockfish_depths:
            results[f'sf_agreement_depth{depth}'] = (results[f'sf_agreement_depth{depth}'] / n) * 100
            results[f'model_in_sf_top3_depth{depth}'] = (results[f'model_in_sf_top3_depth{depth}'] / n) * 100
            results[f'model_in_sf_top5_depth{depth}'] = (results[f'model_in_sf_top5_depth{depth}'] / n) * 100
            results[f'sf_in_model_top3_depth{depth}'] = (results[f'sf_in_model_top3_depth{depth}'] / n) * 100
            results[f'sf_in_model_top5_depth{depth}'] = (results[f'sf_in_model_top5_depth{depth}'] / n) * 100
    
    return results


def main():
    print("="*70)
    print("COMPREHENSIVE CHESS MODEL EVALUATION - LOWER ELO STOCKFISH")
    print("Testing all models at depths [5, 8, 10] (~2000-2400 ELO)")
    print("="*70)
    
    base_dir = Path('/workspace/6s890-finalproject')
    output_dir = base_dir / 'experiments' / 'scripts' / 'eval_results_low_elo'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # All models use same depths now
    experiments = {
        'baseline': {
            'config': base_dir / 'experiments/configs/baseline_config.py',
            'checkpoint': base_dir / 'experiments/results/baseline_mixed_skill/checkpoints/best_baseline_mixed_skill.pt',
            'data_folder': base_dir / 'data',
            'h5_file': 'all_chunks_combined.h5'
        },
        'expert': {
            'config': base_dir / 'experiments/configs/expert_config.py',
            'checkpoint': base_dir / 'experiments/results/expert_LE22ct/checkpoints/best_expert_LE22ct.pt',
            'data_folder': base_dir / 'data/expert',
            'h5_file': 'LE22ct.h5'
        },
        'game_theoretic_qre': {
            'config': base_dir / 'experiments/configs/game_theoretic_qre_config.py',
            'checkpoint': base_dir / 'experiments/results/game_theoretic_qre/checkpoints/best_game_theoretic_qre.pt',
            'data_folder': base_dir / 'data/expert',
            'h5_file': 'LE22ct.h5'
        }
    }
    
    all_results = {}
    
    for exp_name, exp_config in experiments.items():
        print(f"\n{'='*70}")
        print(f"EVALUATING: {exp_name.upper()}")
        print(f"{'='*70}")
        
        # Check if files exist
        if not Path(exp_config['config']).exists():
            print(f"  ✗ Config not found: {exp_config['config']}")
            continue
        if not Path(exp_config['checkpoint']).exists():
            print(f"  ✗ Checkpoint not found: {exp_config['checkpoint']}")
            continue
        
        model, config, metadata = load_model_and_config(
            str(exp_config['config']), 
            str(exp_config['checkpoint'])
        )
        print(f"✓ Model loaded (step: {metadata.get('step', 'unknown')})")
        
        # Try different split options
        dataset = None
        for split_option in ['val', 'test', None]:
            try:
                dataset = ChessDataset(
                    data_folder=str(exp_config['data_folder']),
                    h5_file=exp_config['h5_file'],
                    split=split_option,
                    n_moves=1
                )
                print(f"  ✓ Dataset loaded with split='{split_option}' ({len(dataset)} positions)")
                break
            except Exception as e:
                if split_option is None:
                    print(f"  ERROR: Could not load dataset: {e}")
                    continue
                continue
        
        if dataset is None:
            print(f"  ERROR: Could not load dataset for {exp_name}")
            continue
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # All models use depths [5, 8, 10]
        results = comprehensive_evaluation(
            model, 
            dataloader, 
            n_samples=1000, 
            stockfish_depths=[5, 8, 10]
        )
        results['metadata'] = metadata
        results['experiment'] = exp_name
        all_results[exp_name] = results
        
        # Save
        output_file = output_dir / f'{exp_name}_eval.json'
        with open(output_file, 'w') as f:
            save_results = {k: v for k, v in results.items() 
                          if not k.endswith('_correct') and not k.endswith('_total')}
            json.dump(save_results, f, indent=2)
        print(f"✓ Saved: {output_file}")
    
    # Save comparison
    if len(all_results) > 0:
        comparison_file = output_dir / 'comparison.json'
        with open(comparison_file, 'w') as f:
            comparison = {}
            for exp_name, results in all_results.items():
                comparison[exp_name] = {k: v for k, v in results.items() 
                                       if not k.endswith('_correct') and not k.endswith('_total')}
            json.dump(comparison, f, indent=2)
        print(f"✓ Saved comparison: {comparison_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    for exp_name, results in all_results.items():
        print(f"\n{exp_name.upper()}:")
        print(f"  Samples: {results['n_evaluated']}")
        print(f"  Legal Move Rate: {results['legal_move_rate']:.2f}%")
        print(f"  Top-1: {results['top1_accuracy']:.2f}%")
        print(f"  Top-5: {results['top5_accuracy']:.2f}%")
        print(f"  Stockfish Agreement:")
        for depth in [5, 8, 10]:
            print(f"    Depth {depth}: {results[f'sf_agreement_depth{depth}']:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
