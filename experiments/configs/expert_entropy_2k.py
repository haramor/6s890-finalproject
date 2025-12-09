"""
Expert Configuration - Entropy Regularization (2000 steps)

Quick training run on expert dataset with entropy regularization to encourage
exploration and reduce overconfidence.
"""

import pathlib

# Experiment name
NAME = "expert_entropy_2k"
EXPERIMENT_TYPE = "expert_entropy"

###############################
############ Paths ############
###############################

BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_FOLDER = "/workspace/6s890-finalproject/data/expert"
CHECKPOINT_FOLDER = str(BASE_DIR / "results" / NAME / "checkpoints")
LOGS_FOLDER = str(BASE_DIR / "results" / NAME / "logs")
EVAL_GAMES_FOLDER = str(BASE_DIR / "results" / NAME / "eval_games")

###############################
######### Dataloading #########
###############################

BATCH_SIZE = 512
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
PIN_MEMORY = True

# Dataset configuration
H5_FILE = "LE22ct.h5"
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

# Training configuration - FAST RUN
BATCHES_PER_STEP = 4
PRINT_FREQUENCY = 50  # Print every 50 steps
N_STEPS = 2000  # Quick 2K steps
WARMUP_STEPS = 200  # Proportionally reduced warmup
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
EVAL_FREQUENCY = 200  # Evaluate every 200 steps (10 evals total)

# Checkpoint saving
SAVE_FREQUENCY = 500  # Save every 500 steps
CHECKPOINT_AVG_SUFFIX = ".pt"
TRAINING_CHECKPOINT = None

###############################
######## Entropy Reg ##########
###############################

# Entropy regularization settings
USE_ENTROPY_REGULARIZATION = True
ENTROPY_WEIGHT = 0.01  # λ parameter - encourages exploration
                        # Higher values = more exploration/less confidence
                        # Try 0.001, 0.01, 0.05 for different levels

###############################
####### Game-Theoretic ########
###############################

# Not using GT regularization in this run
USE_GT_REGULARIZATION = False
GT_WEIGHT = 0.0
STOCKFISH_DEPTH = None
