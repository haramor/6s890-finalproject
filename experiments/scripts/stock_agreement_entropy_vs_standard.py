#!/usr/bin/env python3
"""
Stockfish Agreement Comparison: Expert Entropy vs Expert Standard
=================================================================

Compares checkpoint performance (Top-1/Top-3 accuracy + Stockfish agreement)
across two experiment result folders:

  experiments/results/expert_entropy_2k/
  experiments/results/expert_standard_2k/

Assumes each has:
  checkpoints/
  logs/

Uses Stockfish Elo limiter (UCI_LimitStrength + UCI_Elo).
If your Stockfish build enforces an Elo minimum (e.g. 1320), requested Elo values
are automatically clamped into the supported [min, max] range.

Outputs:
  - JSON per experiment
  - Combined JSON
  - Terminal formatted arrays + CSV (easy to plot)

Usage:
  python stock_agreement_entropy_vs_standard.py
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

# --- Project imports ----------------------------------------------------------
sys.path.insert(0, "/workspace/6s890-finalproject/chess-transformers")

from chess_transformers.transformers.models import ChessTransformerEncoder
from chess_transformers.train.datasets import ChessDataset
from chess_transformers.data.levels import UCI_MOVES

# --- Config ------------------------------------------------------------------
BASE_DIR = Path("/workspace/6s890-finalproject")
RESULTS_ROOT = BASE_DIR / "experiments" / "results"
CONFIGS_DIR = BASE_DIR / "experiments" / "configs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STOCKFISH_PATH = "/usr/games/stockfish"

# Requested Elo levels (will be clamped if Stockfish enforces min/max)
REQUESTED_ELOS = [1000, 1500, 2000, 2500]

# Fixed search budget for stockfish move selection
STOCKFISH_DEPTH = 10
STOCKFISH_LIMIT = chess.engine.Limit(depth=STOCKFISH_DEPTH)

# Dataset defaults (if your expert runs used a different folder/file, edit here)
DEFAULT_DATA_FOLDER = str(BASE_DIR / "data" / "expert")
DEFAULT_H5_FILE = "LE22ct.h5"

# Reverse mappings
MOVE_INDEX_TO_UCI = {v: k for k, v in UCI_MOVES.items()}
UCI_TO_INDEX = UCI_MOVES.copy()


# --- Utils: config discovery --------------------------------------------------
def find_config_for_experiment(exp_name: str) -> Path:
    """
    Tries to find a config file under experiments/configs that matches the exp name.
    If your config filenames differ, edit the patterns below.
    """
    patterns = [
        f"*{exp_name}*.py",
        f"*{exp_name.replace('_2k','')}*.py",
        f"*{exp_name.split('_2k')[0]}*.py",
        # common shorthand patterns
        "*entropy*2k*.py" if "entropy" in exp_name else "",
        "*standard*2k*.py" if "standard" in exp_name else "",
        "*expert*entropy*.py" if "entropy" in exp_name else "",
        "*expert*standard*.py" if "standard" in exp_name else "",
    ]
    patterns = [p for p in patterns if p]

    matches: List[Path] = []
    for pat in patterns:
        matches.extend(list(CONFIGS_DIR.glob(pat)))

    # Prefer "expert_entropy_2k_config.py" style if present
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(
            f"Could not find a config for '{exp_name}' in {CONFIGS_DIR}.\n"
            f"Looked for patterns: {patterns}\n"
            f"Fix: either rename your config file to include '{exp_name}', or edit find_config_for_experiment()."
        )
    return matches[0]


# --- Utils: checkpoint discovery ---------------------------------------------
def extract_step_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"step[_-]?(\d+)", path.stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


def infer_checkpoint_sort_key(ckpt_file: Path) -> Tuple[int, str]:
    """
    Returns (key, source). key is ideally training step; falls back to mtime.
    Keeps 'best_*.pt' style checkpoints included.
    """
    step = extract_step_from_filename(ckpt_file)
    if step is not None:
        return step, "filename"

    # try metadata keys on CPU
    try:
        ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for k in ("step", "global_step", "trainer/global_step", "iteration", "iter"):
                v = ckpt.get(k, None)
                if isinstance(v, (int, np.integer)):
                    return int(v), f"ckpt[{k}]"
    except Exception:
        pass

    return int(ckpt_file.stat().st_mtime), "mtime"


def find_checkpoints(checkpoint_dir: Path) -> List[Tuple[int, str, Path]]:
    checkpoint_dir = Path(checkpoint_dir)
    candidates: List[Path] = []

    if checkpoint_dir.exists():
        for pat in ("*.pt", "*.pth", "*.ckpt"):
            candidates.extend(list(checkpoint_dir.glob(pat)))

    # If user passed the experiment root dir by mistake, try /checkpoints
    if not candidates and (checkpoint_dir / "checkpoints").exists():
        for pat in ("*.pt", "*.pth", "*.ckpt"):
            candidates.extend(list((checkpoint_dir / "checkpoints").glob(pat)))

    out: List[Tuple[int, str, Path]] = []
    for f in candidates:
        key, src = infer_checkpoint_sort_key(f)
        out.append((key, src, f))
    out.sort(key=lambda x: x[0])
    return out


# --- Model loading ------------------------------------------------------------
def load_model_and_config(config_path: Path, checkpoint_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", str(config_path))
    cfg = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cfg)

    model = ChessTransformerEncoder(cfg)
    ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        meta = {"epoch": ckpt.get("epoch", "unknown"), "step": ckpt.get("step", "unknown")}
    else:
        model.load_state_dict(ckpt)
        meta = {}

    model.to(DEVICE)
    model.eval()
    return model, cfg, meta


# --- Board decoding -----------------------------------------------------------
def decode_board_position(encoded_board):
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
    boards: List[chess.Board] = []
    bs = batch["board_positions"].shape[0]

    for i in range(bs):
        enc = batch["board_positions"][i].cpu().numpy()
        board = decode_board_position(enc)

        turn_code = batch["turns"][i].item()
        board.turn = chess.WHITE if turn_code == 1 else chess.BLACK

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


# --- Stockfish Elo handling ---------------------------------------------------
def get_uci_elo_range(engine: chess.engine.SimpleEngine) -> Optional[Tuple[int, int]]:
    opt = None
    try:
        opt = engine.protocol.options.get("UCI_Elo")
    except Exception:
        opt = None
    if opt is None:
        return None
    lo = getattr(opt, "min", None)
    hi = getattr(opt, "max", None)
    if lo is None or hi is None:
        return None
    return int(lo), int(hi)


def clamp_elos_to_engine(elos: List[int]) -> Tuple[List[int], Optional[Tuple[int, int]]]:
    tmp = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    try:
        try:
            tmp.isready()
        except Exception:
            pass
        rng = get_uci_elo_range(tmp)
        if rng is None:
            return elos, None
        lo, hi = rng
        clamped = [max(lo, min(hi, int(e))) for e in elos]
        return clamped, (lo, hi)
    finally:
        try:
            tmp.quit()
        except Exception:
            pass


def configure_stockfish_for_elo(engine: chess.engine.SimpleEngine, elo: int) -> None:
    try:
        engine.isready()
    except Exception:
        pass

    # If this throws, it will be a real unsupported-option issue (not min/max),
    # because we clamp elos before using them.
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(elo)})

    try:
        engine.ucinewgame()
        engine.isready()
    except Exception:
        pass


def get_stockfish_move(board: chess.Board, engine: chess.engine.SimpleEngine) -> Optional[str]:
    try:
        r = engine.play(board, STOCKFISH_LIMIT)
        if r and r.move:
            return r.move.uci()
    except Exception:
        return None
    return None


# --- Evaluation ---------------------------------------------------------------
def evaluate_checkpoint(model, dataloader, elos: List[int], n_samples: int = 500) -> Optional[Dict]:
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

        processed = 0
        model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                if processed >= n_samples:
                    break

                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)

                preds = model(batch)          # [B, T, V]
                logits = preds[:, 0, :]       # [B, V]
                vocab_size = logits.shape[-1]
                gt_moves = batch["moves"][:, 1]

                try:
                    boards = batch_to_board_states(batch)
                except Exception:
                    continue

                for i, board in enumerate(boards):
                    if processed >= n_samples:
                        break

                    if board is None or not board.is_valid():
                        continue

                    legal = list(board.legal_moves)
                    if not legal:
                        continue
                    legal_uci = {m.uci() for m in legal}

                    mask = torch.full((vocab_size,), float("-inf"), device=logits.device)
                    for m in legal:
                        idx = UCI_TO_INDEX.get(m.uci(), None)
                        if idx is not None:
                            mask[idx] = 0.0

                    masked = logits[i] + mask
                    topk = min(3, len(legal))
                    top_idx = torch.topk(masked, topk).indices.detach().cpu().numpy()

                    model_top1 = MOVE_INDEX_TO_UCI.get(int(top_idx[0]), None)
                    model_top3 = [MOVE_INDEX_TO_UCI.get(int(j), None) for j in top_idx]
                    gt_uci = MOVE_INDEX_TO_UCI.get(int(gt_moves[i].item()), None)

                    if model_top1 is None:
                        continue

                    if model_top1 in legal_uci:
                        counters["legal_moves"] += 1

                    if gt_uci is not None and model_top1 == gt_uci:
                        counters["top1_correct"] += 1
                    if gt_uci is not None and gt_uci in model_top3:
                        counters["top3_correct"] += 1

                    for elo in elos:
                        sf_move = get_stockfish_move(board, engines[int(elo)])
                        if sf_move is not None and model_top1 == sf_move:
                            counters["sf_agreement_by_elo_count"][int(elo)] += 1

                    counters["n_evaluated"] += 1
                    processed += 1

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
    # clamp elos to engine range so 1000 won't crash on min=1320 builds
    elos, elo_rng = clamp_elos_to_engine(REQUESTED_ELOS)

    print("=" * 80)
    print("STOCKFISH AGREEMENT: EXPERT ENTROPY vs EXPERT STANDARD")
    print(f"Stockfish depth limit: {STOCKFISH_DEPTH}")
    print(f"Requested Elo levels: {REQUESTED_ELOS}")
    if elo_rng is not None:
        print(f"Stockfish UCI_Elo supported range: {elo_rng[0]}..{elo_rng[1]}")
        if elos != REQUESTED_ELOS:
            print(f"Using clamped Elo levels: {elos}")
    else:
        print(f"Could not read UCI_Elo min/max; using requested Elo levels as-is: {elos}")
    print("=" * 80)

    output_dir = BASE_DIR / "experiments" / "scripts" / "training_progression_results" / "entropy_vs_standard"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = {
        "expert_entropy_2k": {
            "result_dir": RESULTS_ROOT / "expert_entropy_2k",
            "checkpoint_dir": RESULTS_ROOT / "expert_entropy_2k" / "checkpoints",
            "data_folder": DEFAULT_DATA_FOLDER,
            "h5_file": DEFAULT_H5_FILE,
        },
        "expert_standard_2k": {
            "result_dir": RESULTS_ROOT / "expert_standard_2k",
            "checkpoint_dir": RESULTS_ROOT / "expert_standard_2k" / "checkpoints",
            "data_folder": DEFAULT_DATA_FOLDER,
            "h5_file": DEFAULT_H5_FILE,
        },
    }

    all_results: Dict[str, List[Dict]] = {}

    # Load dataset (once) since both runs should share expert data
    dataset = None
    for split_option in ["val", "test", None]:
        try:
            dataset = ChessDataset(
                data_folder=DEFAULT_DATA_FOLDER,
                h5_file=DEFAULT_H5_FILE,
                split=split_option,
                n_moves=1,
            )
            print(f"✓ Dataset loaded from {DEFAULT_DATA_FOLDER}/{DEFAULT_H5_FILE} with split='{split_option}'")
            break
        except Exception:
            continue

    if dataset is None:
        print("✗ Could not load dataset. Edit DEFAULT_DATA_FOLDER / DEFAULT_H5_FILE at top of file.")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    for exp_name, cfg in experiments.items():
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT: {exp_name.upper()}")
        print(f"{'=' * 80}")

        # config file discovery
        try:
            config_path = find_config_for_experiment(exp_name)
            print(f"  ✓ Using config: {config_path.name}")
        except Exception as e:
            print(f"  ✗ {e}")
            all_results[exp_name] = []
            continue

        # checkpoints
        ckpts = find_checkpoints(cfg["checkpoint_dir"])
        if not ckpts:
            print(f"  ✗ No checkpoints found in {cfg['checkpoint_dir']}")
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

        exp_rows: List[Dict] = []
        for key, src, ckpt_path in selected:
            print(f"\n  Evaluating: {ckpt_path.name} (key={key}, src={src})")

            try:
                model, _, _ = load_model_and_config(config_path, ckpt_path)
            except Exception as e:
                print(f"    ✗ Failed to load model: {e}")
                continue

            res = evaluate_checkpoint(model, dataloader, elos=elos, n_samples=500)
            if res is None:
                print("    ✗ Evaluation returned no samples (n=0)")
                continue

            res["step_key"] = int(key)
            res["step_key_source"] = src
            res["checkpoint"] = ckpt_path.name
            exp_rows.append(res)

            sf_bits = " | ".join([f"Elo {elo}: {res['sf_agreement_by_elo_pct'][elo]:.2f}%" for elo in elos])
            print(f"    {sf_bits} | Top1={res['top1_accuracy']:.2f}% | Top3={res['top3_accuracy']:.2f}%")

        all_results[exp_name] = exp_rows

        out_file = output_dir / f"{exp_name}_progression_elo.json"
        with open(out_file, "w") as f:
            json.dump(exp_rows, f, indent=2)
        print(f"\n  ✓ Saved: {out_file}")

    combined_file = output_dir / "entropy_vs_standard_all_progressions_elo.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved combined results: {combined_file}")

    # --- Formatted output for plotting ---------------------------------------
    print(f"\n{'=' * 80}")
    print("FORMATTED OUTPUT FOR PLOTTING")
    print(f"{'=' * 80}\n")

    for exp_name, rows in all_results.items():
        if not rows:
            continue
        print(f"{exp_name.upper()}:")
        print(f"  Step keys: {[r['step_key'] for r in rows]}")
        print(f"  Step key source: {[r['step_key_source'] for r in rows]}")
        for elo in elos:
            arr = [round(r["sf_agreement_by_elo_pct"][elo], 2) for r in rows]
            print(f"  SF Agreement @Elo {elo} (%): {arr}")
        print(f"  Top1 Accuracy (%): {[round(r['top1_accuracy'], 2) for r in rows]}")
        print(f"  Top3 Accuracy (%): {[round(r['top3_accuracy'], 2) for r in rows]}")
        print()

    # CSV
    print("\nCSV FORMAT:")
    header = ["experiment", "step_key", "step_key_source", "checkpoint", "top1_accuracy", "top3_accuracy"] + [
        f"sf_agreement_elo_{elo}" for elo in elos
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
            ] + [f"{r['sf_agreement_by_elo_pct'][elo]:.2f}" for elo in elos]
            print(",".join(row))

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
