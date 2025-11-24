import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any


def merge_counts(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] += int(v)


def load_chunk_stats(stats_path: Path) -> dict:
    try:
        return json.loads(stats_path.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to read {stats_path}: {e}") from e


def combine_stats(chunks_dir: Path) -> dict:
    """
    Combine per-chunk stats.json files into one global stats dict.
    Expects stats.json with keys:
      - total_games, white_wins, black_wins, draws, decisive_games
      - decisive_fraction
      - total_half_moves, winner_half_moves, winner_half_fraction
      - elo_all, elo_winner, elo_loser (dicts of bucket -> count)
    """

    stats_files = sorted(chunks_dir.glob("chunk_*/stats.json"))
    if not stats_files:
        raise FileNotFoundError(f"No stats.json found under {chunks_dir}/chunk_*/")

    print(f"[combine_stats] Found {len(stats_files)} stats.json files:")
    for p in stats_files:
        print(f"  - {p}")

    # Global accumulators
    total_chunks = 0
    total_games = 0
    white_wins = 0
    black_wins = 0
    draws = 0
    decisive_games = 0
    total_half_moves = 0
    winner_half_moves = 0

    elo_all = defaultdict(int)
    elo_winner = defaultdict(int)
    elo_loser = defaultdict(int)

    for stats_path in stats_files:
        s = load_chunk_stats(stats_path)
        total_chunks += 1

        total_games += int(s.get("total_games", 0))
        white_wins += int(s.get("white_wins", 0))
        black_wins += int(s.get("black_wins", 0))
        draws += int(s.get("draws", 0))
        decisive_games += int(s.get("decisive_games", 0))

        total_half_moves += int(s.get("total_half_moves", 0))
        winner_half_moves += int(s.get("winner_half_moves", 0))

        merge_counts(elo_all, s.get("elo_all", {}))
        merge_counts(elo_winner, s.get("elo_winner", {}))
        merge_counts(elo_loser, s.get("elo_loser", {}))

    decisive_fraction = (
        decisive_games / total_games if total_games > 0 else 0.0
    )
    winner_half_fraction = (
        winner_half_moves / total_half_moves if total_half_moves > 0 else 0.0
    )

    combined = {
        "total_chunks": total_chunks,
        "total_games": total_games,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "decisive_games": decisive_games,
        "decisive_fraction": decisive_fraction,
        "total_half_moves": total_half_moves,
        "winner_half_moves": winner_half_moves,
        "winner_half_fraction": winner_half_fraction,
        "elo_all": dict(elo_all),
        "elo_winner": dict(elo_winner),
        "elo_loser": dict(elo_loser),
    }

    return combined


def print_hist(title: str, counts: Dict[str, int]) -> None:
    print(title)
    if not counts:
        print("  (no data)")
        return

    for bucket in sorted(counts.keys()):
        print(f"  {bucket:>8}: {counts[bucket]}")
    print()


def maybe_plot_histograms(combined: dict, plot_prefix: str | None) -> None:
    if not plot_prefix:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[combine_stats] matplotlib not installed; skipping plots.")
        return

    def plot_one(name: str, counts: Dict[str, int]) -> None:
        if not counts:
            return
        buckets = sorted(counts.keys())
        values = [counts[b] for b in buckets]

        plt.figure()
        plt.bar(buckets, values)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out = f"{plot_prefix}_{name}.png"
        plt.savefig(out)
        plt.close()
        print(f"[combine_stats] Saved {name} histogram to {out}")

    plot_one("elo_all", combined.get("elo_all", {}))
    plot_one("elo_winner", combined.get("elo_winner", {}))
    plot_one("elo_loser", combined.get("elo_loser", {}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks-dir",
        type=str,
        required=True,
        help="Directory containing chunk_*/stats.json",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save combined stats as JSON.",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default=None,
        help="Optional prefix for saving PNG histograms (requires matplotlib).",
    )
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    combined = combine_stats(chunks_dir)

    # Pretty print summary
    print("\n=== GLOBAL DATASET SUMMARY ===")
    print(f"Chunks combined: {combined['total_chunks']}")
    print(f"Total games    : {combined['total_games']}")
    print(f"  White wins   : {combined['white_wins']}")
    print(f"  Black wins   : {combined['black_wins']}")
    print(f"  Draws        : {combined['draws']}")
    print(
        f"  Decisive     : {combined['decisive_games']} "
        f"({combined['decisive_fraction']:.2%})"
    )
    print(f"Total half-moves (plies): {combined['total_half_moves']}")
    print(
        f"Winner half-moves       : {combined['winner_half_moves']} "
        f"({combined['winner_half_fraction']:.4f})"
    )
    print()

    print_hist("=== Elo Histogram: All Players ===", combined["elo_all"])
    print_hist("=== Elo Histogram: Winners ===", combined["elo_winner"])
    print_hist("=== Elo Histogram: Losers ===", combined["elo_loser"])

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(combined, indent=2))
        print(f"[combine_stats] Saved combined stats JSON to {out_path}")

    maybe_plot_histograms(combined, args.plot_prefix)


if __name__ == "__main__":
    main()
