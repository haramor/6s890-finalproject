# Create New Data

This section explains how to set up dependencies, download raw chess game data, and convert it into filtered `.pgn`, `.fens`, `.moves`, and `.h5` files ready for model training.

---

## 1. Environment Setup

Before starting, ensure your Python environment and dependencies are correctly configured.

### Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# The next line may only work for gpu enabled machines. For mac maybe try torch==2.2.2
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e ./chess-transformers --no-deps # Maybe not no-deps dending on your machine
cd chess-transformers
make
```

## 2. Download Data

From either [Data-1](https://database.nikonoel.fr/) or [Data-2](https://database.lichess.org/)

If you get it from Data-2 you need to unzip the pgn.zst and you should use the following command

```bash
unzstd file_name.pgn.zst
```

## 3. Arrange Files Properly

Put the pngs in a subfolder (Ex. pgn_files)

Move the build_dataset.sh into that folder

Put tags.txt into your folder and add the conditions for filtering that you want for example. The file can have the following conditions

WhiteElo >= "2100"
BlackElo >= "2000"
Elo >= "2400" # This means that at least one player hits this threshold
EloDiff <= "200"
TimeControl >= "300" # This means the game lasts at lest 300 seconds


## 4. Build the .fens and .moves files

Then in build_dataset.sh right after the third pgn-extract (in the command) add --checkmate if you want to filter by only checkmate games

For only decisive games ues -Tr1-0 -Tr0-1

To filter by the winners moves only use 

```bash
pgn-extract -T result=1-0 input.pgn -o white_wins.pgn
pgn-extract -T result=0-1 input.pgn -o black_wins.pgn
```

To keep the .pgn created after filering for analyzation delete ```bash rm -f filtered_games.pgn ``` from build file.

If this is your first time running this
run chmod +x build_dataset.sh in your subfolder

Then run ./build_dataset.sh

Find more tag information [here](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/help.html)

## 5. Create the .h5 file

Then in .venv2/lib/python3.11/site-packages/chess_transformers/configs/data/custom_config.py edit the data_folder to point to your folder, the name of the output file, and optionally other parameters

Then run prepare_data.py

## 6. Analyze games 

Run analyze_games.py and fill in the filtered path

I want stats of number of players at each elo in 100 ranges
I want number of games
Number of half moves total and half moves made by winner
