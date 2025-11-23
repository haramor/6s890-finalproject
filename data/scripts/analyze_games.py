# scripts/analyze_games.py

import argparse
import json
from collections import defaultdict
from pathlib import Path

import chess.pgn
import pandas as pd


def analyze_pgn(pgn_path: Path, out_prefix: Path):
    """
    Analyze a filtered PGN:
      - ELO distribution
      - total games, half-moves, winner half-moves

    Writes:
      - <out_prefix>_summary.json
      - <out_prefix>_elo.csv
    """
    elo_bins = list(range(800, 2900, 100))
    elo_counts = defaultdict(int)
    game_count = 0
    total_half_moves = 0
    winner_half_moves = 0

    def bucket_elo(elo: int) -> str:
        for b in elo_bins:
            if int(elo) < b + 100:
                return f"{b}-{b+99}"
        return f"{elo_bins[-1]}+"

    with pgn_path.open(encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_count += 1

            white_elo = game.headers.get("WhiteElo")
            black_elo = game.headers.get("BlackElo")
            result = game.headers.get("Result")

            for e in (white_elo, black_elo):
                if e and e.isdigit():
                    elo_counts[bucket_elo(int(e))] += 1

            moves = list(game.mainline_moves())
            n_moves = len(moves)
            total_half_moves += n_moves

            if result == "1-0":
                winner_half_moves += (n_moves + 1) // 2
            elif result == "0-1":
                winner_half_moves += n_moves // 2

    elo_df = pd.DataFrame(
        {"Elo Range": list(elo_counts.keys()), "Num Players": list(elo_counts.values())}
    ).sort_values("Elo Range")

    print("=== Dataset Summary ===")
    print(f"Total games: {game_count}")
    print(f"Total half-moves: {total_half_moves}")
    print(f"Half-moves by winner: {winner_half_moves}\n")

    if not elo_df.empty:
        print("=== Elo Distribution (per 100-point range) ===")
        print(elo_df.to_string(index=False))

    summary = {
        "total_games": game_count,
        "total_half_moves": total_half_moves,
        "winner_half_moves": winner_half_moves,
    }
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (out_prefix.with_suffix("_summary.json")).write_text(json.dumps(summary, indent=2))
    elo_df.to_csv(out_prefix.with_suffix("_elo.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=str, required=True, help="Path to filtered PGN.")
    parser.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Prefix for output files (without extension).",
    )
    args = parser.parse_args()

    analyze_pgn(Path(args.pgn), Path(args.out_prefix))


if __name__ == "__main__":
    main()
