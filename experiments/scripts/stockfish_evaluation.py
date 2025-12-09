"""
Stockfish Evaluation Script
Compares model predictions against Stockfish recommendations.
"""

import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add chess-transformers to path
sys.path.insert(0, '/workspace/6s890-finalproject/chess-transformers')

from chess_transformers.transformers.models import ChessTransformerEncoder
from chess_transformers.train.datasets import ChessDataset
from chess_transformers.data.levels import UCI_MOVES, PIECES, TURN, BOOL

# Import chess libraries
import chess
import chess.engine

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STOCKFISH_PATH = '/usr/games/stockfish'

# Create reverse mappings
MOVE_INDEX_TO_UCI = {v: k for k, v in UCI_MOVES.items()}
INDEX_TO_PIECE = {v: k for k, v in PIECES.items()}
INDEX_TO_TURN = {v: k for k, v in TURN.items()}

NEG_INF = -1e9


def load_model_and_config(config_path, checkpoint_path):
    """Load model and config - using the exact method from safe_evaluate.py that worked."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    model = ChessTransformerEncoder(config)

    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(
            f"✓ Checkpoint loaded (epoch: {checkpoint.get('epoch', 'unknown')}, step: {checkpoint.get('step', 'unknown')})"
        )
    else:
        model.load_state_dict(checkpoint)
        print("✓ Checkpoint loaded")

    model.eval()
    return model, config


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
            # Dataset: rank 0=8th, rank 7=1st (top to bottom)
            # python-chess: rank 0=1st, rank 7=8th (bottom to top)
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

        # Set turn
        turn_code = batch['turns'][i].item()
        board.turn = chess.WHITE if turn_code == 1 else chess.BLACK

        # Set castling rights
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


def decode_move_index(move_idx):
    """Convert move index back to UCI move string."""
    return MOVE_INDEX_TO_UCI.get(move_idx, None)


def _legal_mask_for_board(board, vocab_size, device):
    """
    Returns a (vocab_size,) tensor with 0 for legal moves, -inf for illegal/unmapped moves.
    """
    mask = torch.full((vocab_size,), NEG_INF, device=device)

    # Allow only legal moves that exist in UCI_MOVES
    for mv in board.legal_moves:
        uci = mv.uci()
        idx = UCI_MOVES.get(uci, None)
        if idx is not None and 0 <= idx < vocab_size:
            mask[idx] = 0.0

    return mask


def evaluate_against_stockfish(model, dataloader, n_samples=500, stockfish_depth=10):
    print(f"\n{'='*70}")
    print(f"Evaluating {n_samples} positions against Stockfish (depth {stockfish_depth})")
    print(f"{'='*70}")

    model.to(DEVICE)
    model.eval()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # We’ll count Stockfish metrics over positions where:
    # - board decoded ok
    # - we can produce at least one legal move in-vocab (mask has some 0s)
    results = {
        'n_samples': 0,            # total items seen (capped by n_samples)
        'n_eval': 0,               # actually evaluated positions used for %s
        'top1_accuracy': 0,        # masked top-1 == ground truth (count)
        'legal_move_rate': 0,      # masked top-1 legal+in-vocab (count); should match n_eval
        'stockfish_agreement_d5': 0,
        'stockfish_agreement_d10': 0,
        'stockfish_agreement_d15': 0,
        'stockfish_top3_d10': 0,
        'stockfish_top5_d10': 0,
        'model_top3_contains_stockfish': 0,
        'model_top5_contains_stockfish': 0,
        'skipped_no_legal_in_vocab': 0,
        'skipped_bad_board': 0,
    }

    samples_processed = 0
    printed_sanity = False

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if samples_processed >= n_samples:
                    break

                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(DEVICE)

                batch_size = batch['moves'].shape[0]

                # Model predictions
                predictions = model(batch)              # (B, n_moves, V)
                pred_moves = predictions[:, 0, :]        # (B, V)
                vocab_size = pred_moves.shape[-1]

                # Ground truth move index
                actual_move = batch['moves'][:, 1]       # (B,)

                # Decode boards
                try:
                    boards = batch_to_board_states(batch)
                except Exception as e:
                    print(f"\nWarning: Could not decode boards in batch {batch_idx}: {e}")
                    results['skipped_bad_board'] += batch_size
                    samples_processed += batch_size
                    results['n_samples'] = samples_processed
                    continue

                # Optional sanity print once
                if not printed_sanity:
                    gt_legal = 0
                    gt_total = 0
                    for i in range(min(batch_size, len(boards))):
                        b = boards[i]
                        gt_idx = actual_move[i].item()
                        gt_uci = decode_move_index(gt_idx)
                        if not gt_uci:
                            continue
                        try:
                            mv = chess.Move.from_uci(gt_uci)
                        except Exception:
                            continue
                        gt_total += 1
                        if mv in b.legal_moves:
                            gt_legal += 1
                    print(f"\n[Sanity] GT legal rate on decoded boards: {gt_legal}/{gt_total} = {100*gt_legal/max(1,gt_total):.1f}%")
                    print(f"[Sanity] pred vocab = {vocab_size}, len(UCI_MOVES) = {len(UCI_MOVES)}")
                    printed_sanity = True

                # Evaluate each position
                for i in range(min(batch_size, len(boards))):
                    if samples_processed >= n_samples:
                        break

                    board = boards[i]
                    if board is None or not board.is_valid():
                        results['skipped_bad_board'] += 1
                        samples_processed += 1
                        continue

                    # Build legal mask (0 for legal+mapped moves, -inf otherwise)
                    mask = _legal_mask_for_board(board, vocab_size, pred_moves.device)

                    # If there are no legal moves that map into the vocab, skip
                    if torch.all(mask <= (NEG_INF / 2)):
                        results['skipped_no_legal_in_vocab'] += 1
                        samples_processed += 1
                        continue

                    # Mask logits and take top-k among legal moves
                    masked_logits = pred_moves[i] + mask

                    top1_idx = int(torch.argmax(masked_logits).item())
                    top3_idx = torch.topk(masked_logits, k=3).indices.detach().cpu().numpy().tolist()
                    top5_idx = torch.topk(masked_logits, k=5).indices.detach().cpu().numpy().tolist()

                    # top1 should now always be legal+mapped
                    model_move_uci = decode_move_index(top1_idx)
                    if model_move_uci is None:
                        # Should be rare; treat as skip
                        results['skipped_no_legal_in_vocab'] += 1
                        samples_processed += 1
                        continue

                    try:
                        model_move = chess.Move.from_uci(model_move_uci)
                    except Exception:
                        results['skipped_no_legal_in_vocab'] += 1
                        samples_processed += 1
                        continue

                    if model_move not in board.legal_moves:
                        # Also should be rare now (mapping/promotion edge)
                        results['skipped_no_legal_in_vocab'] += 1
                        samples_processed += 1
                        continue

                    results['n_eval'] += 1
                    results['legal_move_rate'] += 1

                    # Accuracy vs ground truth (still exact next-move match)
                    if top1_idx == int(actual_move[i].item()):
                        results['top1_accuracy'] += 1

                    # Stockfish analysis
                    try:
                        sf_result_d5 = engine.analyse(board, chess.engine.Limit(depth=5))
                        sf_move_d5 = sf_result_d5['pv'][0]

                        sf_result_d10 = engine.analyse(board, chess.engine.Limit(depth=10), multipv=5)
                        sf_move_d10 = sf_result_d10[0]['pv'][0]
                        sf_top3_d10 = [r['pv'][0] for r in sf_result_d10[:min(3, len(sf_result_d10))]]
                        sf_top5_d10 = [r['pv'][0] for r in sf_result_d10[:min(5, len(sf_result_d10))]]

                        sf_result_d15 = engine.analyse(board, chess.engine.Limit(depth=15))
                        sf_move_d15 = sf_result_d15['pv'][0]

                        if model_move == sf_move_d5:
                            results['stockfish_agreement_d5'] += 1
                        if model_move == sf_move_d10:
                            results['stockfish_agreement_d10'] += 1
                        if model_move == sf_move_d15:
                            results['stockfish_agreement_d15'] += 1

                        if model_move in sf_top3_d10:
                            results['stockfish_top3_d10'] += 1
                        if model_move in sf_top5_d10:
                            results['stockfish_top5_d10'] += 1

                        # Build model top-3/top-5 move objects (legal by construction, but re-check)
                        model_top3_moves = []
                        model_top5_moves = []

                        for idx in top3_idx:
                            uci = decode_move_index(int(idx))
                            if not uci:
                                continue
                            try:
                                mv = chess.Move.from_uci(uci)
                            except Exception:
                                continue
                            if mv in board.legal_moves:
                                model_top3_moves.append(mv)

                        for idx in top5_idx:
                            uci = decode_move_index(int(idx))
                            if not uci:
                                continue
                            try:
                                mv = chess.Move.from_uci(uci)
                            except Exception:
                                continue
                            if mv in board.legal_moves:
                                model_top5_moves.append(mv)

                        if sf_move_d10 in model_top3_moves:
                            results['model_top3_contains_stockfish'] += 1
                        if sf_move_d10 in model_top5_moves:
                            results['model_top5_contains_stockfish'] += 1

                    except Exception as e:
                        # If stockfish fails, still count it as evaluated position for accuracy/legal rate
                        # (You can change this behavior if you want.)
                        pass

                    samples_processed += 1

                results['n_samples'] = samples_processed

    finally:
        engine.quit()

    # Convert counts to percentages (denominator = n_eval)
    n_eval = results['n_eval']
    if n_eval > 0:
        results['top1_accuracy'] = (results['top1_accuracy'] / n_eval) * 100.0
        results['legal_move_rate'] = (results['legal_move_rate'] / n_eval) * 100.0

        for k in [
            'stockfish_agreement_d5',
            'stockfish_agreement_d10',
            'stockfish_agreement_d15',
            'stockfish_top3_d10',
            'stockfish_top5_d10',
            'model_top3_contains_stockfish',
            'model_top5_contains_stockfish',
        ]:
            results[k] = (results[k] / n_eval) * 100.0
    else:
        # Avoid division by zero
        results['top1_accuracy'] = 0.0
        results['legal_move_rate'] = 0.0
        for k in [
            'stockfish_agreement_d5',
            'stockfish_agreement_d10',
            'stockfish_agreement_d15',
            'stockfish_top3_d10',
            'stockfish_top5_d10',
            'model_top3_contains_stockfish',
            'model_top5_contains_stockfish',
        ]:
            results[k] = 0.0

    return results


def main():
    print("=" * 70)
    print("STOCKFISH EVALUATION")
    print("=" * 70)

    output_dir = Path('/workspace/6s890-finalproject/results/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(STOCKFISH_PATH).exists():
        print(f"ERROR: Stockfish not found at {STOCKFISH_PATH}")
        print("Please install: apt-get update && apt-get install -y stockfish")
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

        print("Loading model...")
        model, config = load_model_and_config(exp_config['config'], exp_config['checkpoint'])
        print("✓ Model loaded")

        print("Loading dataset...")
        dataset = ChessDataset(
            data_folder=exp_config['data_folder'],
            h5_file=exp_config['h5_file'],
            split='val',
            n_moves=1
        )

        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )
        print(f"✓ Dataset loaded ({len(dataset)} positions)")

        results = evaluate_against_stockfish(
            model,
            dataloader,
            n_samples=1000,
            stockfish_depth=10
        )

        all_results[exp_name] = results

        output_file = output_dir / f'{exp_name}_stockfish_eval.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved: {output_file}")

    comparison_file = output_dir / 'stockfish_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved comparison: {comparison_file}")

    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY (LEGAL-MOVE MASKED)")
    print(f"{'='*70}")

    for exp_name, results in all_results.items():
        print(f"\n{exp_name.upper()}:")
        print(f"  Samples Seen: {results.get('n_samples', 0)}")
        print(f"  Positions Evaluated (n_eval): {results.get('n_eval', 0)}")
        print(f"  Skipped (no legal in vocab): {results.get('skipped_no_legal_in_vocab', 0)}")
        print(f"  Skipped (bad board): {results.get('skipped_bad_board', 0)}")
        print(f"  Legal Move Rate (masked): {results.get('legal_move_rate', 0):.2f}%")
        print(f"  Top-1 Accuracy (vs ground truth, masked): {results.get('top1_accuracy', 0):.2f}%")
        print(f"  Stockfish Agreement (depth 5): {results.get('stockfish_agreement_d5', 0):.2f}%")
        print(f"  Stockfish Agreement (depth 10): {results.get('stockfish_agreement_d10', 0):.2f}%")
        print(f"  Stockfish Agreement (depth 15): {results.get('stockfish_agreement_d15', 0):.2f}%")
        print(f"  Model in Stockfish Top-3 (d10): {results.get('stockfish_top3_d10', 0):.2f}%")
        print(f"  Model in Stockfish Top-5 (d10): {results.get('stockfish_top5_d10', 0):.2f}%")
        print(f"  Stockfish in Model Top-3 (d10): {results.get('model_top3_contains_stockfish', 0):.2f}%")
        print(f"  Stockfish in Model Top-5 (d10): {results.get('model_top5_contains_stockfish', 0):.2f}%")

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
