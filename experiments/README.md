# Chess Behavioral Cloning with Game-Theoretic Rationality

This directory contains experiments for the 6.S890 final project on leveraging game-theoretic rationality to reduce sample complexity in behavioral cloning for chess.

## Project Structure

```
experiments/
├── configs/          # Configuration files for different experimental conditions
├── data/            # Data preparation scripts and sample datasets
├── models/          # Custom model implementations and loss functions
├── results/         # Training logs, checkpoints, and evaluation results
├── scripts/         # Training and evaluation scripts
└── README.md        # This file
```

## Experimental Conditions

### 1. Baseline (Mixed Skill)
- **Config**: `configs/baseline_config.py`
- **Dataset**: Mixed skill levels (ELO 1500-2500+)
- **Purpose**: Establish performance with standard behavioral cloning

### 2. Expert-Only
- **Config**: `configs/expert_only_config.py`
- **Dataset**: High-ELO players only (2500+)
- **Hypothesis**: 30-50% reduction in sample complexity due to lower variance

### 3. Game-Theoretic Regularization
- **Config**: `configs/game_theoretic_config.py`
- **Dataset**: High-ELO players (2500+)
- **Loss**: Cross-entropy + KL-divergence from Stockfish evaluations
- **Hypothesis**: Additional 15-25% reduction in sample complexity

## Small-Scale Test Setup

For the progress report, we're using a minimal setup:
- **Model**: CT-E-20 architecture (20M parameters, encoder-only)
- **Training data**: ~10K games
- **Evaluation**: Top-1 and top-3 move accuracy on held-out test set

## Key Metrics

1. **Sample Complexity Curves**: Training games required to achieve 70% top-1 accuracy
2. **Top-K Accuracy**: Top-1, top-3, and top-5 move prediction accuracy
3. **Stockfish Alignment**: KL-divergence between model and Stockfish move distributions
4. **Centipawn Loss**: Average position evaluation change from predicted moves

## Dependencies

Core dependencies (from chess-transformers):
- PyTorch
- python-chess (for Stockfish integration)
- tensorboard (for logging)
- h5py/tables (for HDF5 data files)

## Quick Start

1. **Data Preparation**: See `data/prepare_sample_data.py`
2. **Training**: Run `python scripts/train.py --config configs/baseline_config.py`
3. **Evaluation**: Run `python scripts/evaluate.py --checkpoint results/baseline/checkpoint.pt`

## Notes

- Install Stockfish separately and set `CT_STOCKFISH_PATH` environment variable
- For full-scale experiments, use the Lichess database from https://database.lichess.org/
