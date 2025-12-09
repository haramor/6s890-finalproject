#!/usr/bin/env python3
"""
Stockfish Agreement Over Training (Elo-Limited)
===============================================

Evaluates model checkpoints at different training steps to see how Stockfish
agreement evolves during training.

Key differences vs depth-only scripts:
- Uses Stockfish's Elo limiter (UCI_LimitStrength + UCI_Elo) and compares the
  model's Top-1 move to Stockfish's chosen move at Elo {1000, 1500, 2000, 2500}.
- Uses engine.play(...) (the move Stockfish would actually play at that strength),
  not engine.analyse(...).

Usage:
    python stock_agreement_elo_progression.py

Outputs:
    - One JSON per experiment
    - Combined JSON across experiments
    - Terminal output formatted for plotting (arrays + CSV)
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import chess
import chess.engine

# --- Paths / Imports (project-specific) ---------------------------------------
# Add chess-transformers to path
sys.path.insert(0, "/workspace/6s890-finalproject/chess-transformers")

from chess_transformers.transformers.models import ChessTransformerEncoder
from chess_transformers.train.datasets import ChessDataset
from chess_transformers.data.levels import UCI_MOVES

# --- Config ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STOCKFISH_PATH = "/usr/games/stockfish"

# Elo levels to test
ELO_TESTS = [1500, 2000, 2500]

# Fixed search budget. Depth is acceptable; movetime is also fine.
# If you prefer time-based: chess.engine.Limit(time=0.05) or chess.engine.Limit(nodes=...)
STOCKFISH_DEPTH = 10
STOCKFISH_LIMIT = chess.engine.Limit(depth=STOCKFISH_DEPTH)

# Reverse mappings
MOVE_INDEX_TO_UCI = {v: k for k, v in UCI_MOVES.items()}
UCI_TO_INDEX = UCI_MOVES.copy()


# --- Checkpoint helpers -------------------------------------------------------
def extract_step_from_checkpoint_filename(checkpoint_path: Path) -> Optional[int]:
    """
    Extract step number from checkpoint filename like:
      checkpoint_step_12345.pt
      ckpt-step12345.pt
      ...step_123...
    """
    filename = checkpoint_path.stem
    match = re.search(r"step[_-]?(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else None


def infer_checkpoint_sort_key(ckpt_file: Path) -> Tuple[int, str]:
    """
    Returns (key, source). key is ideally training step, otherwise falls back to
    a stable ordering key (mtime) so step-less checkpoints like best_*.pt are included.
    """
    # 1) From filename
    step = extract_step_from_checkpoint_filename(ckpt_file)
    if step is not None:
        return step, "filename"

    # 2) From checkpoint metadata (common patterns)
    try:
        ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for k in ("step", "global_step", "trainer/global_step", "iteration", "iter"):
                v = ckpt.get(k, None)
                if isinstance(v, (int, np.integer)):
                    return int(v), f"ckpt[{k}]"
    except Exception:
        pass

    # 3) Fallback: file modified time (monotone-ish)
    return int(ckpt_file.stat().st_mtime), "mtime"


def find_checkpoints(checkpoint_dir: Path) -> List[Tuple[int, str, Path]]:
    """
    Find checkpoints and return sorted list of (sort_key, key_source, path).
    Includes step-less .pt files (e.g., best_random_skill.pt).
    """
    checkpoint_dir = Path(checkpoint_dir)
    candidates: List[Path] = []

    # direct directory scan
    if checkpoint_dir.exists():
        for pattern in ("*.pt", "*.pth", "*.ckpt"):
            candidates.extend(list(checkpoint_dir.glob(pattern)))

    # if none, search recursively one level up (handles slightly different layouts)
    if not candidates and checkpoint_dir.parent.exists():
        for pattern in ("*.pt", "*.pth", "*.ckpt"):
            candidates.extend(list(checkpoint_dir.parent.rglob(pattern)))

    out: List[Tuple[int, str, Path]] = []
    for f in candidates:
        key, src = infer_checkpoint_sort_key(f)
        out.append((key, src, f))

    out.sort(key=lambda x: x[0])
    return out


# --- Model loading ------------------------------------------------------------
def load_model_and_config(config_path: str, checkpoint_path: str):
    """Load model and config."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(config)

    model = ChessTransformerEncoder(config)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    metadata = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        metadata = {"epoch": checkpoint.get("epoch", "unknown"), "step": checkpoint.get("step", "unknown")}
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model, config, metadata


# --- Board decoding -----------------------------------------------------------
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


def batch_to_board_states(batch) -> List[chess.Board]:
    """Convert batch data to list of chess boards."""
    boards: List[chess.Board] = []
    batch_size = batch["board_positions"].shape[0]

    for i in range(batch_size):
        board_encoded = batch["board_positions"][i].cpu().numpy()
        board = decode_board_position(board_encoded)

        turn_code = batch["turns"][i].item()
        board.turn = chess.WHITE if turn_code == 1 else chess.BLACK

        # Castling rights: dataset stores booleans; python-chess uses bitboards.
        board.castling_rights = 0
        if batch["white_kingside_castling_rights"][i].item() == 1:
            board.castling_rights |= chess.BB_H1
        if batch["white_queenside_castling_rights"][i].item() == 1:
            board.castling_rights |= chess.BB_A1
        if batch["black_kingside_castling_rights"][i].item() == 1:
            board.castling_rights |= chess.BB_H8
        if batch["black_queenside_castling_rights"][i].item() == 1:
            board.castling_rights |= chess.BB_A8

        boards.append(board)

    return boards


# --- Stockfish helpers --------------------------------------------------------
def configure_stockfish_for_elo(engine: chess.engine.SimpleEngine, elo: int) -> None:
    """Configure Stockfish to play at an approximate target Elo."""
    try:
        engine.isready()
    except Exception:
        pass

    try:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(elo)})
    except Exception as e:
        raise RuntimeError(
            "Your Stockfish build does not accept UCI_LimitStrength/UCI_Elo. "
            f"Original error: {e}"
        )

    try:
        engine.ucinewgame()
        engine.isready()
    except Exception:
        pass


def get_stockfish_move(board: chess.Board, engine: chess.engine.SimpleEngine, limit: chess.engine.Limit) -> Optional[str]:
    """Get the move Stockfish would actually play at its current configured strength."""
    try:
        r = engine.play(board, limit)
        if r and r.move:
            return r.move.uci()
    except Exception:
        return None
    return None


# --- Evaluation ---------------------------------------------------------------
def evaluate_checkpoint(model, dataloader, n_samples: int = 500, elos: List[int] = None) -> Optional[Dict]:
    """Evaluate Top-1/Top-3 plus Stockfish agreement per Elo for one checkpoint."""
    if elos is None:
        elos = ELO_TESTS

    # One engine per Elo is simplest and avoids option carryover.
    engines: Dict[int, chess.engine.SimpleEngine] = {}
    try:
        for elo in elos:
            eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            configure_stockfish_for_elo(eng, elo)
            engines[int(elo)] = eng

        counters = {
            "n_evaluated": 0,
            "legal_moves": 0,
            "top1_correct": 0,
            "top3_correct": 0,
            "sf_agreement_by_elo_count": {int(elo): 0 for elo in elos},
        }

        samples_processed = 0
        model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                if samples_processed >= n_samples:
                    break

                # move all tensors
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)

                preds = model(batch)                # [B, T, V] (your usage is preds[:,0,:])
                pred_moves = preds[:, 0, :]         # [B, V]
                vocab_size = pred_moves.shape[-1]
                actual_moves = batch["moves"][:, 1] # ground truth move idx

                try:
                    boards = batch_to_board_states(batch)
                except Exception:
                    continue

                for i, board in enumerate(boards):
                    if samples_processed >= n_samples:
                        break

                    if board is None or not board.is_valid():
                        continue

                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        continue
                    legal_uci = {m.uci() for m in legal_moves}

                    # Mask illegal moves (prevents illegal top-k)
                    mask = torch.full((vocab_size,), float("-inf"), device=pred_moves.device)
                    for m in legal_moves:
                        idx = UCI_TO_INDEX.get(m.uci(), None)
                        if idx is not None:
                            mask[idx] = 0.0

                    masked_logits = pred_moves[i] + mask
                    topk = min(3, len(legal_moves))
                    top_indices = torch.topk(masked_logits, topk).indices.detach().cpu().numpy()

                    model_top1 = MOVE_INDEX_TO_UCI.get(int(top_indices[0]), None)
                    model_top3 = [MOVE_INDEX_TO_UCI.get(int(j), None) for j in top_indices]
                    gt_uci = MOVE_INDEX_TO_UCI.get(int(actual_moves[i].item()), None)

                    if model_top1 is None:
                        continue

                    if model_top1 in legal_uci:
                        counters["legal_moves"] += 1

                    if gt_uci is not None and model_top1 == gt_uci:
                        counters["top1_correct"] += 1
                    if gt_uci is not None and gt_uci in model_top3:
                        counters["top3_correct"] += 1

                    # Stockfish agreement per Elo
                    for elo in elos:
                        sf_move = get_stockfish_move(board, engines[int(elo)], STOCKFISH_LIMIT)
                        if sf_move is not None and model_top1 == sf_move:
                            counters["sf_agreement_by_elo_count"][int(elo)] += 1

                    counters["n_evaluated"] += 1
                    samples_processed += 1

        n = counters["n_evaluated"]
        if n <= 0:
            return None

        sf_pct = {int(elo): (counters["sf_agreement_by_elo_count"][int(elo)] / n) * 100 for elo in elos}

        return {
            "n_samples": n,
            "legal_move_rate": (counters["legal_moves"] / n) * 100,
            "top1_accuracy": (counters["top1_correct"] / n) * 100,
            "top3_accuracy": (counters["top3_correct"] / n) * 100,
            "sf_depth": STOCKFISH_DEPTH,
            "sf_agreement_by_elo_pct": sf_pct,
            "sf_agreement_by_elo_count": counters["sf_agreement_by_elo_count"],
        }

    finally:
        for eng in engines.values():
            try:
                eng.quit()
            except Exception:
                pass


# --- Main --------------------------------------------------------------------
def main():
    print("=" * 80)
    print("STOCKFISH AGREEMENT OVER TRAINING (ELO-LIMITED)")
    print(f"Stockfish depth limit: {STOCKFISH_DEPTH}")
    print(f"Testing Elo levels: {ELO_TESTS}")
    print("=" * 80)

    base_dir = Path("/workspace/6s890-finalproject")
    output_dir = base_dir / "experiments" / "scripts" / "training_progression_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = {
        "baseline": {
            "config": base_dir / "experiments/configs/baseline_config.py",
            "checkpoint_dir": base_dir / "experiments/results/baseline_mixed_skill/checkpoints",
            "data_folder": base_dir / "data",
            "h5_file": "all_chunks_combined.h5",
        },
        "expert": {
            "config": base_dir / "experiments/configs/expert_config.py",
            "checkpoint_dir": base_dir / "experiments/results/expert_LE22ct/checkpoints",
            "data_folder": base_dir / "data/expert",
            "h5_file": "LE22ct.h5",
        },
        "random": {
            "config": base_dir / "experiments/configs/random_config.py",
            "checkpoint_dir": base_dir / "experiments/results/random_skill/checkpoints",
            "data_folder": base_dir / "data",
            "h5_file": "rand_chunks_combined.h5",
        },
    }

    all_results: Dict[str, List[Dict]] = {}

    for exp_name, exp_cfg in experiments.items():
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT: {exp_name.upper()}")
        print(f"{'=' * 80}")

        ckpts = find_checkpoints(exp_cfg["checkpoint_dir"])
        if not ckpts:
            print(f"  ✗ No checkpoints found in {exp_cfg['checkpoint_dir']}")
            all_results[exp_name] = []
            continue

        print(f"  Found {len(ckpts)} checkpoint file(s)")

        # Choose up to 10 evenly spaced checkpoints
        if len(ckpts) <= 10:
            selected = ckpts
        else:
            idxs = np.linspace(0, len(ckpts) - 1, 10, dtype=int)
            selected = [ckpts[i] for i in idxs]

        print(f"  Evaluating {len(selected)} checkpoints:")
        for key, src, p in selected:
            print(f"    - Key {key} ({src}): {p.name}")

        # Load dataset once
        dataset = None
        for split_option in ["val", "test", None]:
            try:
                dataset = ChessDataset(
                    data_folder=str(exp_cfg["data_folder"]),
                    h5_file=exp_cfg["h5_file"],
                    split=split_option,
                    n_moves=1,
                )
                print(f"  ✓ Dataset loaded with split='{split_option}'")
                break
            except Exception:
                continue

        if dataset is None:
            print("  ✗ Could not load dataset")
            all_results[exp_name] = []
            continue

        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

        exp_results: List[Dict] = []

        for key, src, ckpt_path in selected:
            print(f"\n  Evaluating checkpoint: {ckpt_path.name} (key={key}, src={src})")
            try:
                model, config, metadata = load_model_and_config(str(exp_cfg["config"]), str(ckpt_path))
            except Exception as e:
                print(f"    ✗ Failed to load model: {e}")
                continue

            res = evaluate_checkpoint(model, dataloader, n_samples=500, elos=ELO_TESTS)
            if res is None:
                print("    ✗ Evaluation returned no samples (n=0)")
                continue

            # Store whatever "key" is (step or mtime) so plots are ordered
            res["step_key"] = int(key)
            res["step_key_source"] = src
            res["checkpoint"] = ckpt_path.name

            exp_results.append(res)

            sf_bits = " | ".join([f"Elo {elo}: {res['sf_agreement_by_elo_pct'][elo]:.2f}%" for elo in ELO_TESTS])
            print(f"    {sf_bits} | Top1={res['top1_accuracy']:.2f}% | Top3={res['top3_accuracy']:.2f}%")

        all_results[exp_name] = exp_results

        out_file = output_dir / f"{exp_name}_progression_elo.json"
        with open(out_file, "w") as f:
            json.dump(exp_results, f, indent=2)
        print(f"\n  ✓ Saved: {out_file}")

    combined_file = output_dir / "all_progressions_elo.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved combined results: {combined_file}")

    # Formatted output
    print(f"\n{'=' * 80}")
    print("FORMATTED OUTPUT FOR PLOTTING")
    print(f"{'=' * 80}\n")

    for exp_name, rows in all_results.items():
        if not rows:
            continue
        print(f"{exp_name.upper()}:")
        print(f"  Step keys: {[r['step_key'] for r in rows]}")
        print(f"  Step key source: {[r['step_key_source'] for r in rows]}")
        for elo in ELO_TESTS:
            arr = [round(r["sf_agreement_by_elo_pct"][elo], 2) for r in rows]
            print(f"  SF Agreement @Elo {elo} (%): {arr}")
        print(f"  Top1 Accuracy (%): {[round(r['top1_accuracy'], 2) for r in rows]}")
        print(f"  Top3 Accuracy (%): {[round(r['top3_accuracy'], 2) for r in rows]}")
        print()

    # CSV
    print("\nCSV FORMAT:")
    header = ["experiment", "step_key", "step_key_source", "checkpoint", "top1_accuracy", "top3_accuracy"] + [
        f"sf_agreement_elo_{elo}" for elo in ELO_TESTS
    ]
    print(",".join(header))
    for exp_name, rows in all_results.items():
        for r in rows:
            row = [
                exp_name,
                str(r["step_key"]),
                str(r["step_key_source"]),
                str(r["checkpoint"]),
                f"{r['top1_accuracy']:.2f}",
                f"{r['top3_accuracy']:.2f}",
            ] + [f"{r['sf_agreement_by_elo_pct'][elo]:.2f}" for elo in ELO_TESTS]
            print(",".join(row))

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
