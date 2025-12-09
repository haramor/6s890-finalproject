"""
Game-Theoretic Regularization Configuration

This configuration uses expert data (2500+) with additional game-theoretic
regularization that penalizes deviations from Stockfish's evaluation.
"""

import torch
import pathlib
import os

# Experiment name
NAME = "game_theoretic_reg"
EXPERIMENT_TYPE = "game_theoretic"

###############################
############ Paths ############
###############################

BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_FOLDER = str(BASE_DIR / "data" / "expert_2500")
CHECKPOINT_FOLDER = str(BASE_DIR / "results" / NAME / "checkpoints")
LOGS_FOLDER = str(BASE_DIR / "results" / NAME / "logs")
EVAL_GAMES_FOLDER = str(BASE_DIR / "results" / NAME / "eval_games")

# Stockfish path - set via environment variable or default
STOCKFISH_PATH = os.environ.get("CT_STOCKFISH_PATH", "/usr/games/stockfish")

###############################
######### Dataloading #########
###############################

BATCH_SIZE = 512  # Same as other conditions
NUM_WORKERS = 8  # Set to 0 to avoid multiprocessing pickle issues with HDF5
PREFETCH_FACTOR = 2
PIN_MEMORY = True  # Set to False for CPU training

# Dataset configuration
H5_FILE = "expert_2500_10k.h5"
N_MOVES = 1

###############################
############ Model ############
###############################

# Same architecture for fair comparison
VOCAB_SIZES = {
    "moves": 1968,
    "turn": 2,
    "white_kingside_castling_rights": 2,
    "white_queenside_castling_rights": 2,
    "black_kingside_castling_rights": 2,
    "black_queenside_castling_rights": 2,
    "board_position": 13,
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
N_STEPS = 100000
WARMUP_STEPS = 8000
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
# Start with 0.1, tune based on validation performance
GT_WEIGHT = 0.5

# Stockfish configuration for equilibrium oracle
STOCKFISH_DEPTH = 15  # Depth for position evaluation
STOCKFISH_TIME_LIMIT = 0.1  # seconds per position (for efficiency)
STOCKFISH_THREADS = 1

# Cache Stockfish evaluations to avoid recomputation
USE_STOCKFISH_CACHE = True
STOCKFISH_CACHE_SIZE = 10000

# Option: precompute Stockfish evaluations offline
# (set to True for faster training)
PRECOMPUTE_STOCKFISH = False
