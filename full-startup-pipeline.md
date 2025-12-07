# Final Project – Environment & Pipeline Setup (Full Rebuild from Scratch)

This README documents how to rebuild the environment inside a fresh container,  
including installing PyTorch, building + installing local chess-transformers,  
compiling pgn-extract, and preparing everything for the chunked PGN pipeline.

--------------------------------------------------------------------------------
0. Repository Layout (expected)
--------------------------------------------------------------------------------

/workspace/6s890-finalproject
  chess-transformers/
    Makefile
    setup.py
    chess_transformers/
      configs/models/custom_config.py
      data/prep.py
      ...
  scripts/
    build_dataset.sh
    process_chunks.py
    combine_h5.py
    analyze_games.py
    combine_stats.py
    tags.txt
  data/
    data_files/
      <YOUR_HUGE_PGN>.pgn
    chunks/


--------------------------------------------------------------------------------
1. System Dependencies (Ubuntu) -- This can be skipped
--------------------------------------------------------------------------------

apt-get update
apt-get install -y \
    python3-venv python3-dev \
    build-essential \
    wget make \
    zlib1g-dev \
    git \
    curl

--------------------------------------------------------------------------------
2. Create Virtual Environment
--------------------------------------------------------------------------------
cd /workspace
git clone https://github.com/haramor/6s890-finalproject.git

cd /workspace/6s890-finalproject
python3 -m venv .venv
source .venv/bin/activate

--------------------------------------------------------------------------------
3. Install PyTorch (GPU/CPU/Mac)
--------------------------------------------------------------------------------
check which python
which pip 

Both should be 
/workspace/6s890-finalproject/.venv/bin/python
/workspace/6s890-finalproject/.venv/bin/pip
otherwise do python -m pip install ...

# ---- GPU-enabled Linux (A100/T4/V100) ----
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# ---- CPU-only Linux or Mac ----
# pip install torch==2.2.2

--------------------------------------------------------------------------------
4. Install Required Python Packages
--------------------------------------------------------------------------------

# IMPORTANT: numpy and tables must be pinned
pip install "numpy==1.26.4" "tables==3.10.2"

# Additional libraries used by the pipeline
pip install \
    pandas \
    python-chess \
    tqdm \
    matplotlib \
    regex \
    h5py \
    ipython

--------------------------------------------------------------------------------
5. Install Local chess-transformers Package
--------------------------------------------------------------------------------

cd /workspace/6s890-finalproject

git clone https://github.com/sgrvinod/chess-transformers.git

# Editable install so that code updates automatically apply
# NO-DEPS prevents pip from overriding numpy/tables/etc.
<!-- pip install -e ./chess-transformers --no-deps  -->
python -m pip install -e ./chess-transformers --no-deps

# Verify
python -c "import chess_transformers; print(chess_transformers.__file__)"

--------------------------------------------------------------------------------
6. Install pgn-extract (v25-01)
--------------------------------------------------------------------------------

cd /tmp
wget https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/pgn-extract-25-01.tgz
tar xvf pgn-extract-25-01.tgz
cd pgn-extract
make

mkdir -p /workspace/6s890-finalproject/bin
cp pgn-extract /workspace/6s890-finalproject/bin/

# Add to PATH (for this session)
export PATH=/workspace/6s890-finalproject/bin:$PATH

# Add to future shells
echo 'export PATH=/workspace/6s890-finalproject/bin:$PATH' >> ~/.bashrc

# Inside venv 
export PATH=/workspace/6s890-finalproject/bin:$PATH

which pgn-extract
pgn-extract --version

--------------------------------------------------------------------------------
7. Data Setup
--------------------------------------------------------------------------------

cd /workspace/6s890-finalproject
mkdir -p data/data_files data/chunks


# Example download (replace with your PGN URL):
# wget -O data/data_files/lichess_2025-05.pgn.zst https://database.lichess.org/standard/lichess_db_standard_rated_2025-05.pgn.zst
# unzstd data/data_files/*.zst

wget -O file_name link

From either [Data-1](https://database.nikonoel.fr/) or [Data-2](https://database.lichess.org/)

If you get it from Data-2 you need to unzip the pgn.zst and you should use the following command

```bash
unzstd file_name.pgn.zst
```

Your “huge” PGN should end up at:
data/data_files/lichess_db_standard_rated_2025-05.pgn

--------------------------------------------------------------------------------
8. Configure tags.txt
--------------------------------------------------------------------------------

Place your global filtering rules at:
scripts/tags.txt

Example:

[Event "Rated Blitz game"]
WhiteElo >= "2100"
BlackElo >= "2000"
Elo >= "2400" # This means that at least one player hits this threshold
EloDiff <= "200"
TimeControl >= "300" # This means the game lasts at lest 300 seconds


process_chunks.py will copy this file into each chunk directory before
running build_dataset.sh so that filtering is consistent.

--------------------------------------------------------------------------------
9. Run the Full Chunked Pipeline
--------------------------------------------------------------------------------
cp scripts/custom_config.py chess-transformers/chess_transformers/configs/models/
source .venv/bin/activate
export PATH=/workspace/6s890-finalproject/bin:$PATH

python scripts/process_chunksv2.py \
  --huge-pgn data/data_files/lichess_2025-05.pgn \
  --chunks-dir data/chunks \
  --chunk-size 1000 \
  --config-name custom_config

The pipeline will:
  - Split into chunks
  - Build .moves/.fens
  - Run prepare_data to create each chunk's H5
  - Run analyze_games to create stats.json
  - Save progress.json so it can resume after Ctrl+C

--------------------------------------------------------------------------------
10. Combine H5 Files
--------------------------------------------------------------------------------

python scripts/combine_h5.py \
  --chunks-dir data/chunks \
  --output data/chunks_combined.h5

--------------------------------------------------------------------------------
11. Combine Stats Across All Chunks
--------------------------------------------------------------------------------

python scripts/combine_stats.py \
  --chunks-dir data/chunks \
  --output-json data/combined_stats.json \
  --plot-prefix data/elo_hist



--------------------------------------------------------------------------------
This completes the full reproducible setup from a blank container.
--------------------------------------------------------------------------------


When logging back on 
cd 
source .venv/bin/activate