#!/usr/bin/env bash
# Run this *inside* the folder that contains the PGN files you want to process.
# This keeps your original PGNs safe and only deletes the numbered chunk PGNs
# the script creates (1.pgn, 2.pgn, ...).

set -euo pipefail

# 1) Create a file list of the PGNs in the current folder (like the original)
#    (Change the glob on the next line if you want to restrict which files.)
find . -maxdepth 1 -name "*.pgn" | sort > filenames.txt

# 2) Combine all files into one
pgn-extract -ffilenames.txt --output combined.pgn

# 3) Clean up the list (keeps your originals)
rm -f filenames.txt

# 4) Time control filter (>= 5 minutes base time per player)
echo 'TimeControl >= "300"' > tags.txt
pgn-extract -t tags.txt combined.pgn --output time_control_gte_5m.pgn
rm -f combined.pgn 

# 5) Filter games that ended in checkmate
pgn-extract --checkmate time_control_gte_5m.pgn --output filtered_games.pgn
rm -f time_control_gte_5m.pgn

# 6) Shuffle deterministically
python3 -c "import regex, random; random.seed(1234); text = open('filtered_games.pgn','r',encoding='utf-8',errors='replace').read(); games = [g for g in regex.split(r'\n{2,}(?=\[Event)', text) if g.strip()]; random.shuffle(games); print(f'{len(games)} games shuffled.'); open('shuffled_filtered_games.pgn','w').write('\n\n'.join(games));"

# 7) Split into chunks of 500k games (fine for tiny files too; you’ll just get 1.pgn)
pgn-extract -#500000 shuffled_filtered_games.pgn
rm -f filtered_games.pgn shuffled_filtered_games.pgn

# 8) Extract FENs and UCI moves for each chunk
#    (If you only got a single chunk, these will be 1.fens and 1.moves)
for p in [0-9]*.pgn; do
  base="${p%.pgn}"
  pgn-extract -Wfen  "$p" --notags --noresults                    --output "${base}.fens"
  pgn-extract -Wlalg "$p" --notags --nomovenumbers --nochecks -w7 --output "${base}.moves"
done

# 9) Remove only the numbered chunk PGNs we created (NOT your original PGNs)
rm -f [0-9]*.pgn

echo "Done. Look for *.fens and *.moves in: $(pwd)"
