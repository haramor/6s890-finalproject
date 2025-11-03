# Game-Theoretic Behavioral Cloning for Chess

**Course**: 6.S890 Multi-Agent Learning
**Team**: Skyler Pulling, Hara Moraitaki, Isaac (Zack) Duitz

## Project Overview

This project tests whether leveraging game-theoretic rationality can reduce sample complexity in behavioral cloning for chess.

**Hypotheses**:
- **H1**: Expert-only training (ELO 2500+) requires 30-50% fewer samples than mixed-skill training
- **H2**: Adding game-theoretic regularization (KL-divergence from Stockfish) provides an additional 15-25% reduction

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/haramor/6s890-finalproject.git
cd 6s890-finalproject
```

### 2. Set Up Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio scipy h5py tqdm tensorboard python-chess tables

# Clone base models (not included in repo)
git clone https://github.com/sgrvinod/chess-transformers.git
```

### 3. Generate Test Data

```bash
cd experiments
python data/prepare_sample_data.py
```

### 4. Run Demo (verify setup)

```bash
python scripts/demo_training.py
```

Expected: 5 training steps with decreasing loss (~30 seconds)

## Three Experimental Conditions

1. **Baseline** (`configs/baseline_config.py`)
   - Mixed skill dataset (ELO 1500-2500+)
   - Standard cross-entropy loss

2. **Expert-Only** (`configs/expert_only_config.py`)
   - High-ELO only (2500+)
   - Standard cross-entropy loss
   - Tests H1

3. **Game-Theoretic** (`configs/game_theoretic_config.py`)
   - High-ELO only (2500+)
   - Loss: CE + λ * KL(Stockfish || Model)
   - Tests H2

## Project Structure

```
experiments/
├── configs/                    # Experimental configurations
├── models/                     # Game-theoretic loss implementation
├── scripts/                    # Training and testing scripts
└── data/                       # Data preparation utilities

chess-transformers/             # Clone separately (not in repo)
```

## Key Innovation

Custom loss function combining behavioral cloning with equilibrium regularization:

```
L_total = L_CE + λ * L_KL

where:
- L_CE = Cross-entropy (learn from expert moves)
- L_KL = KL-divergence from Stockfish (penalize deviation from equilibrium)
- λ = Tunable weight (default: 0.1)
```

## Current Status

### ✅ Complete
- Three experimental configurations
- Game-theoretic loss implementation
- Full training pipeline with TensorBoard
- Training verified working (loss decreasing)

### 📋 Needed
- Real Lichess data (see TODO.md)
- Sample complexity analysis
- Statistical significance tests

## Running Experiments

### Full Training

```bash
source venv/bin/activate
cd experiments

# Baseline
python scripts/train.py --config configs/baseline_config.py

# Expert-only
python scripts/train.py --config configs/expert_only_config.py

# Game-theoretic (requires Stockfish)
python scripts/train.py --config configs/game_theoretic_config.py
```

### View Logs

```bash
tensorboard --logdir results/
# Open http://localhost:6006
```

## Documentation

- **README.md** (this file) - Project overview and setup
- **DATA-PREP.md** - How to prepare a new dataset
- **TODO.md** - Remaining work and detailed roadmap
- **experiments/README.md** - Experiment-specific details

## Requirements

- Python 3.8+
- PyTorch 2.0+
- python-chess, h5py, tensorboard
- Stockfish (for game-theoretic condition)

## License

Academic research project for 6.S890.
