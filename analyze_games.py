import chess.pgn
import pandas as pd
from collections import defaultdict

# Path to your filtered PGN file
PGN_PATH = "/Users/zackduitz/Desktop/organized/MIT/MIT_5th_year_classes/multi-agent_learning/final_project/pgn_files/test_filtered.pgn"

# Elo bins (100-point ranges)
elo_bins = list(range(800, 2900, 100))
elo_counts = defaultdict(int)
game_count = 0
total_half_moves = 0
winner_half_moves = 0

def bucket_elo(elo):
    if elo is None:
        return None
    for b in elo_bins:
        if int(elo) < b + 100:
            return f"{b}-{b+99}"
    return f"{elo_bins[-1]}+"

with open(PGN_PATH, encoding="utf-8", errors="ignore") as f:
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break
        game_count += 1

        # Extract basic data
        white_elo = game.headers.get("WhiteElo")
        black_elo = game.headers.get("BlackElo")
        result = game.headers.get("Result")

        # Count Elo distribution
        for e in (white_elo, black_elo):
            if e and e.isdigit():
                elo_counts[bucket_elo(int(e))] += 1

        # Count half-moves
        moves = list(game.mainline_moves())
        n_moves = len(moves)
        total_half_moves += n_moves

        # Count half-moves made by the winning player
        if result == "1-0":
            # White won → white moves every even index (0-based)
            winner_half_moves += (n_moves + 1) // 2
        elif result == "0-1":
            # Black won → black moves every odd index
            winner_half_moves += n_moves // 2

# Convert to DataFrame for display
elo_df = pd.DataFrame(
    {"Elo Range": list(elo_counts.keys()), "Num Players": list(elo_counts.values())}
).sort_values("Elo Range")

print("=== Dataset Summary ===")
print(f"Total games: {game_count}")
print(f"Total half-moves: {total_half_moves}")
print(f"Half-moves by winner: {winner_half_moves}\n")

print("=== Elo Distribution (per 100-point range) ===")
print(elo_df.to_string(index=False))
