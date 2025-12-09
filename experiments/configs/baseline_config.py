"""
Baseline Configuration - Mixed Skill Dataset

This configuration uses a mixed-skill dataset (ELO 1500-2500+) to establish
baseline performance for behavioral cloning.
"""

import torch
import pathlib

# Experiment name
NAME = "baseline_mixed_skill"
EXPERIMENT_TYPE = "baseline"

###############################
############ Paths ############
###############################

BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_FOLDER = "/workspace/6s890-finalproject/data"
CHECKPOINT_FOLDER = str(BASE_DIR / "results" / NAME / "checkpoints")
LOGS_FOLDER = str(BASE_DIR / "results" / NAME / "logs")
EVAL_GAMES_FOLDER = str(BASE_DIR / "results" / NAME / "eval_games")

###############################
######### Dataloading #########
###############################

# For small-scale testing, we'll use simplified data loading
BATCH_SIZE = 512  # Smaller for testing
NUM_WORKERS = 8  # Set to 0 toavoid multiprocessing pickle issues with HDF5
PREFETCH_FACTOR = 2
PIN_MEMORY = True  # Set toTrue for CPU training

# Dataset configuration
H5_FILE = "all_chunks_combined.h5"  # 10K games for small-scale test
N_MOVES = 1  # Predict only next move (encoder-only model)

###############################
############ Model ############
###############################

# Vocabulary sizes (from chess-transformers)
VOCAB_SIZES = {
    "moves": 1971,  # All possible UCI moves
    "turn": 2,  # White or black
    "white_kingside_castling_rights": 2,
    "white_queenside_castling_rights": 2,
    "black_kingside_castling_rights": 2,
    "black_queenside_castling_rights": 2,
    "board_position": 14,  # Empty + 6 piece types × 2 colors
}

# Model architecture (CT-E-20 style, encoder-only)
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

# Training configuration
BATCHES_PER_STEP = 4
PRINT_FREQUENCY = 10
N_STEPS = 10000  # Small-scale: 1K steps
WARMUP_STEPS = 700
MAX_LR = 0.1
LR_SCHEDULE = "vaswani"
LR_DECAY = None

# Optimizer
BETAS = (0.9, 0.98)
EPSILON = 1e-9
LABEL_SMOOTHING = 0.1

# Mixed precision training
USE_AMP = True

# Compilation
DISABLE_COMPILATION = False
COMPILATION_MODE = "default"
DYNAMIC_COMPILATION = True

###############################
######### Evaluation ##########
###############################

# Evaluation frequency
EVAL_FREQUENCY = 100  # Evaluate every 100 steps

# Checkpoint saving
SAVE_FREQUENCY = 100
CHECKPOINT_AVG_SUFFIX = ".pt"
TRAINING_CHECKPOINT = None  # Set to checkpoint path to resume

###############################
####### Game-Theoretic ########
###############################

# No game-theoretic regularization for baseline
USE_GT_REGULARIZATION = False
GT_WEIGHT = 0.0
STOCKFISH_DEPTH = None
