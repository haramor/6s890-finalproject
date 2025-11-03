# Getting Started Guide

This guide will help you get the small-scale experiment running for your progress report.

## Overview

Your project tests whether game-theoretic rationality can reduce sample complexity in behavioral cloning for chess. The codebase is set up to run three experimental conditions:

1. **Baseline**: Mixed skill dataset (ELO 1500-2500+)
2. **Expert-Only**: High-ELO dataset (2500+)
3. **Game-Theoretic**: Expert data + KL-divergence regularization from Stockfish

## Quick Start (Small-Scale Test)

### Step 1: Install Dependencies

First, install the required Python packages:

```bash
# Basic dependencies (you likely already have these)
pip3 install torch torchvision
pip3 install numpy scipy
pip3 install tensorboard
pip3 install tqdm
pip3 install h5py

# Chess-specific
pip3 install python-chess

# Optional: Install HDF5 for real data (not needed for synthetic test)
# brew install hdf5  # macOS
# pip3 install tables
```

### Step 2: Install Stockfish (Optional for Baseline/Expert-Only)

Stockfish is only needed for the game-theoretic regularization condition.

```bash
# macOS
brew install stockfish

# Then set environment variable
export CT_STOCKFISH_PATH=/usr/local/bin/stockfish
```

For other platforms, download from: https://stockfishchess.org/download/

### Step 3: Create Synthetic Test Data

Since downloading and processing real chess data takes time, start with synthetic data:

```bash
cd experiments
python3 data/prepare_sample_data.py
```

This creates small test datasets (~1000 positions each) to verify the pipeline works.

### Step 4: Run Small-Scale Training

Test each experimental condition:

```bash
# Baseline (mixed skill)
python3 scripts/train.py --config configs/baseline_config.py

# Expert-only
python3 scripts/train.py --config configs/expert_only_config.py

# Game-theoretic regularization (requires Stockfish)
python3 scripts/train.py --config configs/game_theoretic_config.py
```

**Expected output:**
- Training progress with loss and accuracy metrics
- TensorBoard logs in `results/<experiment_name>/logs/`
- Checkpoints in `results/<experiment_name>/checkpoints/`

### Step 5: Monitor Training

```bash
tensorboard --logdir experiments/results/
```

Then open http://localhost:6006 to view training curves.

## What to Show in Your Progress Report

For the progress report, you can demonstrate:

1. **Infrastructure is Working**:
   - Show that all three experimental conditions run without errors
   - Display TensorBoard screenshots with training curves

2. **Initial Observations**:
   - Compare convergence speed across the three conditions
   - Note any preliminary differences in sample efficiency

3. **Next Steps**:
   - Plan for downloading real Lichess data
   - Outline scaling to full 10K-1M game experiments

## Next Steps: Using Real Data

Once you've verified the pipeline works, scale up to real data:

### 1. Download Lichess Data

```bash
# Download from Lichess Elite Database
# See: https://database.nikonoel.fr/

# Example: Download December 2022 data
wget https://database.lichess.org/lichess_elite_YYYY-MM.pgn.zst
```

### 2. Filter by ELO

Use `pgn-extract` or the online tool at https://database.nikonoel.fr/ to filter:

**Expert data (2500+):**
```bash
pgn-extract --tag "WhiteElo>=2500" --tag "BlackElo>=2500" \
  input.pgn -o expert_2500.pgn
```

**Mixed skill (1500-2500):**
```bash
pgn-extract --tag "WhiteElo>=1500" --tag "WhiteElo<=2500" \
  input.pgn -o mixed_skill.pgn
```

### 3. Convert to HDF5

Use the chess-transformers data preparation tools:

```bash
cd chess-transformers
python -m chess_transformers.data.prep --config your_data_config.py
```

See `chess-transformers/chess_transformers/configs/data/` for examples.

### 4. Update Configs

Edit your experiment configs to point to the real data:

```python
# experiments/configs/expert_only_config.py
H5_FILE = "expert_2500_real.h5"  # Your real data file
N_STEPS = 100000  # Scale up training steps
BATCH_SIZE = 512  # Increase batch size if GPU allows
```

## Troubleshooting

### "Could not import from chess-transformers"

The training script has fallbacks, so basic training should still work. For full functionality:

```bash
cd chess-transformers
pip3 install -e .
```

This requires HDF5 to be installed first (see above).

### "Error loading data"

Make sure you've run the data preparation script:

```bash
cd experiments
python3 data/prepare_sample_data.py
```

### Stockfish not found

For baseline and expert-only experiments, you don't need Stockfish. For game-theoretic:

```bash
# Install Stockfish
brew install stockfish  # macOS

# Set path
export CT_STOCKFISH_PATH=$(which stockfish)

# Or edit the config directly
# experiments/configs/game_theoretic_config.py
STOCKFISH_PATH = "/path/to/stockfish"
```

### Out of memory

Reduce batch size in the config:

```python
# experiments/configs/baseline_config.py
BATCH_SIZE = 32  # Or even smaller
```

## Understanding the Code Structure

```
experiments/
├── configs/                    # Experimental configurations
│   ├── baseline_config.py     # Baseline: mixed skill data
│   ├── expert_only_config.py  # Expert: 2500+ ELO only
│   └── game_theoretic_config.py  # GT: expert + Stockfish KL
│
├── data/                       # Data preparation
│   └── prepare_sample_data.py # Generate synthetic test data
│
├── models/                     # Model implementations
│   └── game_theoretic_loss.py # Custom loss with KL regularization
│
├── scripts/                    # Training and evaluation
│   └── train.py               # Main training script
│
└── results/                    # Outputs (created during training)
    ├── baseline_mixed_skill/
    ├── expert_only_2500/
    └── game_theoretic_reg/
```

## Key Implementation Details

### Game-Theoretic Loss

The key innovation is in `models/game_theoretic_loss.py`:

```python
L_total = L_CE + λ * L_KL

where:
- L_CE: Cross-entropy with expert moves (behavioral cloning)
- L_KL: KL(Stockfish || Model) - penalizes deviation from equilibrium
- λ: Weight balancing the two terms (default: 0.1)
```

### Evaluation Metrics

The code tracks:
- **Top-1/3/5 accuracy**: Move prediction accuracy
- **Loss components**: CE loss and KL loss separately
- **Learning curves**: Logged to TensorBoard

For the full evaluation (not in this demo), you'll also want:
- **Centipawn loss**: Using Stockfish evaluation
- **Win rate**: Against various engine strength levels

## Tips for Your Progress Report

1. **Run all three conditions** with synthetic data to show infrastructure works

2. **Create comparison plots**:
   - Training loss curves (all three on same plot)
   - Top-1 accuracy over steps
   - Convergence speed comparison

3. **Discuss preliminary findings**:
   - Does expert-only converge faster than baseline (even with synthetic data)?
   - Any unexpected behaviors?

4. **Outline scaling plan**:
   - Timeline for downloading real data
   - Computational requirements (GPU hours estimate)
   - Hyperparameter tuning strategy

## Contact and Help

If you run into issues:

1. Check the chess-transformers README: `chess-transformers/README.md`
2. Look at their example configs: `chess-transformers/chess_transformers/configs/`
3. Review their training code: `chess-transformers/chess_transformers/train/train.py`

## References

- **Chess-Transformers**: https://github.com/sgrvinod/chess-transformers
- **Lichess Database**: https://database.lichess.org/
- **Lichess Elite Filter**: https://database.nikonoel.fr/
- **Stockfish**: https://stockfishchess.org/
- **python-chess**: https://python-chess.readthedocs.io/

Good luck with your progress report!
