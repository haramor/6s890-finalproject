import argparse
import json
from collections import defaultdict
from pathlib import Path

import chess.pgn
import pandas as pd


def bucket_elo(elo: int | None, elo_bins: list[int]) -> str | None:
    if elo is None:
        return None
    for b in elo_bins:
        if int(elo) < b + 100:
            return f"{b}-{b+99}"
    return f"{elo_bins[-1]}+"


def analyze_pgn(pgn_path: Path) -> dict:
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")

    elo_bins = list(range(800, 2900, 100))

    # Overall player elo counts (both sides)
    elo_counts_all = defaultdict(int)

    # Elo counts for winners and losers separately
    elo_counts_winner = defaultdict(int)
    elo_counts_loser = defaultdict(int)

    game_count = 0
    total_half_moves = 0
    winner_half_moves = 0

    white_wins = 0
    black_wins = 0
    draws = 0

    with pgn_path.open(encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_count += 1

            headers = game.headers
            white_elo = headers.get("WhiteElo")
            black_elo = headers.get("BlackElo")
            result = headers.get("Result")

            # Result counters
            if result == "1-0":
                white_wins += 1
            elif result == "0-1":
                black_wins += 1
            elif result == "1/2-1/2":
                draws += 1

            # Overall Elo (both players)
            if white_elo and white_elo.isdigit():
                bucket = bucket_elo(int(white_elo), elo_bins)
                if bucket is not None:
                    elo_counts_all[bucket] += 1
            if black_elo and black_elo.isdigit():
                bucket = bucket_elo(int(black_elo), elo_bins)
                if bucket is not None:
                    elo_counts_all[bucket] += 1

            # Half-moves
            moves = list(game.mainline_moves())
            n_moves = len(moves)
            total_half_moves += n_moves

            # Winner/loser elo + winner half-moves
            if result == "1-0":
                # White won, Black lost
                winner_half_moves += (n_moves + 1) // 2

                if white_elo and white_elo.isdigit():
                    bucket = bucket_elo(int(white_elo), elo_bins)
                    if bucket is not None:
                        elo_counts_winner[bucket] += 1

                if black_elo and black_elo.isdigit():
                    bucket = bucket_elo(int(black_elo), elo_bins)
                    if bucket is not None:
                        elo_counts_loser[bucket] += 1

            elif result == "0-1":
                # Black won, White lost
                winner_half_moves += n_moves // 2

                if black_elo and black_elo.isdigit():
                    bucket = bucket_elo(int(black_elo), elo_bins)
                    if bucket is not None:
                        elo_counts_winner[bucket] += 1

                if white_elo and white_elo.isdigit():
                    bucket = bucket_elo(int(white_elo), elo_bins)
                    if bucket is not None:
                        elo_counts_loser[bucket] += 1

            # draws still counted in elo_counts_all, but not in winner/loser

    decisive_games = white_wins + black_wins
    decisive_fraction = decisive_games / game_count if game_count > 0 else 0.0
    winner_half_fraction = (
        winner_half_moves / total_half_moves if total_half_moves > 0 else 0.0
    )

    stats = {
        "pgn_path": str(pgn_path),
        "total_games": game_count,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "decisive_games": decisive_games,
        "decisive_fraction": decisive_fraction,
        "total_half_moves": total_half_moves,
        "winner_half_moves": winner_half_moves,
        "winner_half_fraction": winner_half_fraction,
        "elo_all": dict(elo_counts_all),
        "elo_winner": dict(elo_counts_winner),
        "elo_loser": dict(elo_counts_loser),
    }

    return stats


def print_hist(title: str, counts: dict[str, int]) -> None:
    print(title)
    if not counts:
        print("  (no data)")
        return
    for bucket in sorted(counts.keys()):
        print(f"  {bucket:>8}: {counts[bucket]}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn_path", type=str, help="Path to filtered PGN file.")
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save stats as JSON.",
    )
    args = parser.parse_args()

    stats = analyze_pgn(Path(args.pgn_path))

    # Print summary
    print("=== Dataset Summary ===")
    print(f"PGN file: {stats['pgn_path']}")
    print(f"Total games    : {stats['total_games']}")
    print(f"  White wins   : {stats['white_wins']}")
    print(f"  Black wins   : {stats['black_wins']}")
    print(f"  Draws        : {stats['draws']}")
    print(
        f"  Decisive     : {stats['decisive_games']} "
        f"({stats['decisive_fraction']:.2%})"
    )
    print(f"Total half-moves (plies): {stats['total_half_moves']}")
    print(
        f"Winner half-moves       : {stats['winner_half_moves']} "
        f"({stats['winner_half_fraction']:.4f})"
    )
    print()

    print_hist("=== Elo Histogram: All Players ===", stats["elo_all"])
    print_hist("=== Elo Histogram: Winners ===", stats["elo_winner"])
    print_hist("=== Elo Histogram: Losers ===", stats["elo_loser"])

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.write_text(json.dumps(stats, indent=2))
        print(f"Saved analysis to {out_path}")


if __name__ == "__main__":
    main()
