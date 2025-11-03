# Leveraging Game-Theoretic Rationality to Reduce Sample Complexity in Behavioral Cloning

**Course**: 6.S890 Multi-Agent Learning (Fall 2024)
**Team**: Skyler Pulling, Hara Moraitaki, Isaac (Zack) Duitz

## Overview

This project investigates whether explicitly exploiting game-theoretic structure can dramatically reduce the amount of training data needed for behavioral cloning in chess. We test two main hypotheses:

**H1 (Expert Variance)**: Expert-only training (ELO 2500+) requires 30-50% fewer samples than mixed-skill training.

**H2 (Game-Theoretic Regularization)**: Adding explicit game-theoretic regularization through KL-divergence from Stockfish evaluations provides an additional 15-25% sample complexity reduction.

## Project Structure

```
6s890-final/
├── chess-transformers/      # Base transformer models for chess (cloned)
├── experiments/             # Our custom experimental code
│   ├── configs/            # Three experimental conditions
│   ├── data/               # Data preparation utilities
│   ├── models/             # Game-theoretic loss implementation
│   ├── scripts/            # Training and testing scripts
│   └── results/            # Training outputs (created during training)
├── GETTING_STARTED.md      # Detailed setup instructions
├── PROJECT_SUMMARY.md      # Technical overview
└── README.md               # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip3 install torch numpy scipy h5py tqdm tensorboard python-chess
```

### 2. Run Quick Test

```bash
cd experiments
python3 scripts/quick_test.py
```

This will:
- Check that all dependencies are installed
- Create synthetic test data
- Run a minimal training test
- Verify the setup works

### 3. Run Full Experiments

```bash
# Baseline (mixed skill)
python3 scripts/train.py --config configs/baseline_config.py

# Expert-only
python3 scripts/train.py --config configs/expert_only_config.py

# Game-theoretic regularization
python3 scripts/train.py --config configs/game_theoretic_config.py
```

### 4. View Training Progress

```bash
tensorboard --logdir experiments/results/
```

Open http://localhost:6006 to view training curves.

## Three Experimental Conditions

### 1. Baseline (Mixed Skill)
- **Dataset**: Games from ELO 1500-2500+
- **Loss**: Standard cross-entropy with label smoothing
- **Purpose**: Establish baseline performance

### 2. Expert-Only
- **Dataset**: Games from ELO 2500+ only
- **Loss**: Standard cross-entropy with label smoothing
- **Hypothesis**: Lower variance in expert play → faster convergence

### 3. Game-Theoretic Regularization
- **Dataset**: Games from ELO 2500+ only
- **Loss**: Cross-entropy + λ * KL(Stockfish || Model)
- **Hypothesis**: Explicit equilibrium guidance → further sample reduction

## Key Innovation: Game-Theoretic Loss

The core contribution is a novel loss function that combines behavioral cloning with game-theoretic regularization:

```
L_total = L_CE + λ * L_KL

where:
- L_CE = Label-smoothed cross-entropy (learn from expert moves)
- L_KL = KL-divergence from Stockfish (penalize deviations from equilibrium)
- λ = Regularization weight (tunable hyperparameter)
```

**Implementation**: `experiments/models/game_theoretic_loss.py`

## Current Status

✅ **Complete infrastructure for small-scale testing**
- Three experimental configurations
- Custom game-theoretic loss with Stockfish integration
- Full training pipeline with metrics tracking
- Synthetic data generation for testing
- TensorBoard visualization

⏳ **Next steps**
- Download real Lichess data (10K-1M games)
- Run full-scale experiments
- Analyze sample complexity curves
- Compute statistical significance

## For Your Progress Report

This codebase is ready for a progress report demo. You can:

1. **Show working infrastructure**:
   ```bash
   python3 experiments/scripts/quick_test.py
   ```

2. **Run all three conditions** on synthetic data

3. **Generate comparison plots** from TensorBoard logs

4. **Document preliminary findings** and next steps

See `GETTING_STARTED.md` for detailed instructions.

## Documentation

- **`GETTING_STARTED.md`**: Detailed setup and usage instructions
- **`PROJECT_SUMMARY.md`**: Technical details and architecture overview
- **`experiments/README.md`**: Experiment-specific documentation

## Key Files

| File | Purpose |
|------|---------|
| `experiments/configs/*.py` | Experimental configurations |
| `experiments/models/game_theoretic_loss.py` | Custom loss function |
| `experiments/scripts/train.py` | Main training script |
| `experiments/scripts/quick_test.py` | Quick verification script |
| `experiments/data/prepare_sample_data.py` | Create test data |

## Evaluation Metrics

1. **Sample Complexity**: Training samples needed to reach 70% top-1 accuracy
2. **Top-K Accuracy**: Move prediction accuracy (K=1,3,5)
3. **Stockfish Alignment**: KL-divergence between model and engine
4. **Centipawn Loss**: Average position evaluation change
5. **Convergence Speed**: Training steps to reach target performance

## Technical Approach

### Model Architecture
- Transformer encoder (20M parameters)
- Based on chess-transformers CT-E-20 architecture
- Predicts best next move from board state

### Training
- Mixed precision training (AMP)
- Label smoothing (ε=0.1)
- Vaswani learning rate schedule
- Validation during training
- Early stopping on best validation accuracy

### Game-Theoretic Regularization
- Stockfish as equilibrium oracle (depth 15)
- Multi-PV analysis for move distributions
- Softmax over centipawn evaluations
- LRU caching for efficiency

## Data Sources

For real experiments (not included in this repo):

- **Lichess Elite Database**: https://database.lichess.org/
- **Lichess ELO Filter**: https://database.nikonoel.fr/
- **PGN Mentor**: https://www.pgnmentor.com/

The synthetic test data is sufficient for verifying the pipeline works.

## Requirements

### Software
- Python 3.8+
- PyTorch 2.0+
- python-chess
- h5py
- TensorBoard
- Stockfish (for game-theoretic condition)

### Hardware
- GPU recommended (CPU works but slower)
- 8GB+ RAM for small-scale experiments
- 16GB+ RAM for large-scale experiments

## Repository Credits

This project builds on:
- **chess-transformers** by sgrvinod: https://github.com/sgrvinod/chess-transformers
- **Stockfish** chess engine: https://stockfishchess.org/

## License

This project is for academic research (6.S890 final project).

## Contact

For questions about this implementation:
- Check the documentation in `GETTING_STARTED.md`
- Review code comments in key files
- See chess-transformers original README

## Quick Command Reference

```bash
# Setup
cd experiments
python3 scripts/quick_test.py

# Prepare data (if needed)
python3 data/prepare_sample_data.py

# Train models
python3 scripts/train.py --config configs/baseline_config.py
python3 scripts/train.py --config configs/expert_only_config.py
python3 scripts/train.py --config configs/game_theoretic_config.py

# Monitor training
tensorboard --logdir results/

# View logs
ls results/*/logs/
ls results/*/checkpoints/
```

Good luck with your progress report and final project!
