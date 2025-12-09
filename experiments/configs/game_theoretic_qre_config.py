"""
Game-Theoretic Regularization Configuration

This configuration implements game-theoretic regularization based on classical
chess evaluation and bounded rationality concepts from game theory.

GAME-THEORETIC APPROACH:
------------------------
1. Minimax Approximation: Each move is evaluated using:
   - Material balance (zero-sum game payoffs)
   - Positional heuristics (strategic value)
   - Opponent response strength (shallow lookahead)

2. Quantal Response Equilibrium (QRE): Move evaluations are converted to
   probabilities via softmax, modeling bounded rational players who are more
   likely to choose better moves but don't always play optimally.

3. Nash Equilibrium Regularization: The KL-divergence term encourages the
   learned policy to approximate this game-theoretic equilibrium, preventing
   overfitting to specific training trajectories.

THEORETICAL JUSTIFICATION:
-------------------------
- In two-player zero-sum games (like chess), Nash equilibrium = minimax solution
- Expert demonstrations may include suboptimal moves or style biases
- Regularizing toward equilibrium strategies should improve robustness
- QRE provides a more realistic model of human play than pure Nash

LOSS FUNCTION:
--------------
L = L_CE(expert moves) + λ × KL(QRE_distribution || model_distribution)

Where:
- L_CE encourages imitation of expert moves
- KL term encourages game-theoretically sound play
- λ balances imitation vs. equilibrium-seeking behavior
"""

import torch
import pathlib
import os

# Experiment name
NAME = "game_theoretic_qre"  # Quantal Response Equilibrium
EXPERIMENT_TYPE = "game_theoretic"

###############################
############ Paths ############
###############################

BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_FOLDER = str(BASE_DIR.parent / "data" / "expert")  # Changed: go up one more level
CHECKPOINT_FOLDER = str(BASE_DIR / "results" / NAME / "checkpoints")
LOGS_FOLDER = str(BASE_DIR / "results" / NAME / "logs")
EVAL_GAMES_FOLDER = str(BASE_DIR / "results" / NAME / "eval_games")

# Stockfish configuration (optional - only if you have precomputed cache)
# STOCKFISH_CACHE_PATH = str(BASE_DIR / "data" / "expert_2500" / "stockfish_cache.pkl")
# If the above file exists, it will be used. Otherwise, heuristic evaluation is used.

###############################
######### Dataloading #########
###############################

BATCH_SIZE = 512
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
PIN_MEMORY = True

# Dataset configuration
H5_FILE = "LE22ct.h5"  # Full expert dataset (large!)
N_MOVES = 1

###############################
############ Model ############
###############################

VOCAB_SIZES = {
    "moves": 1971,  # All possible UCI moves (LE22ct.h5 uses 1971, not 1968)
    "turn": 2,
    "white_kingside_castling_rights": 2,
    "white_queenside_castling_rights": 2,
    "black_kingside_castling_rights": 2,
    "black_queenside_castling_rights": 2,
    "board_position": 14,  # Empty + 6 piece types × 2 colors (LE22ct.h5 uses 14, not 13)
}

D_MODEL = 512
N_HEADS = 8
D_QUERIES = 64
D_VALUES = 64
D_INNER = 2048
N_LAYERS = 6
DROPOUT = 0.1

###############################
########### Training ##########
###############################

BATCHES_PER_STEP = 1
PRINT_FREQUENCY = 10
N_STEPS = 20000  # Reduced to fit in 24 hours (~11-12 hours training)
WARMUP_STEPS = 2000  # Proportionally reduced
LR_SCHEDULE = "vaswani"
LR_DECAY = None

BETAS = (0.9, 0.98)
EPSILON = 1e-9
LABEL_SMOOTHING = 0.1

USE_AMP = True

DISABLE_COMPILATION = False
COMPILATION_MODE = "default"
DYNAMIC_COMPILATION = True

###############################
######### Evaluation ##########
###############################

EVAL_FREQUENCY = 100
SAVE_FREQUENCY = 100
CHECKPOINT_AVG_SUFFIX = ".pt"
TRAINING_CHECKPOINT = None

###############################
####### Game-Theoretic ########
###############################

# Enable game-theoretic regularization
USE_GT_REGULARIZATION = True

# Weight for KL-divergence term (λ in loss = CE + λ * KL)
GT_WEIGHT = 0.1

# QRE temperature parameter (controls rationality)
# Lower = more deterministic (closer to pure minimax)
# Higher = more exploratory (softer distribution)
QRE_TEMPERATURE = 100.0

# No Stockfish configuration needed - using heuristic evaluation
