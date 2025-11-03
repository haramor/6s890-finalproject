"""
Expert-Only Configuration

This configuration uses only high-ELO games (2500+) to test whether
expert data alone reduces sample complexity.
"""

import torch
import pathlib

# Experiment name
NAME = "expert_only_2500"
EXPERIMENT_TYPE = "expert_only"

###############################
############ Paths ############
###############################

BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_FOLDER = str(BASE_DIR / "data" / "expert_2500")
CHECKPOINT_FOLDER = str(BASE_DIR / "results" / NAME / "checkpoints")
LOGS_FOLDER = str(BASE_DIR / "results" / NAME / "logs")
EVAL_GAMES_FOLDER = str(BASE_DIR / "results" / NAME / "eval_games")

###############################
######### Dataloading #########
###############################

BATCH_SIZE = 64
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing pickle issues with HDF5
PREFETCH_FACTOR = 2
PIN_MEMORY = False  # Set to False for CPU training

# Dataset configuration
H5_FILE = "expert_2500_10k.h5"  # 10K expert games for small-scale test
N_MOVES = 1

###############################
############ Model ############
###############################

# Same model architecture as baseline for fair comparison
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

BATCHES_PER_STEP = 4
PRINT_FREQUENCY = 10
N_STEPS = 1000
WARMUP_STEPS = 100
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

# No game-theoretic regularization (testing expert data only)
USE_GT_REGULARIZATION = False
GT_WEIGHT = 0.0
STOCKFISH_DEPTH = None
