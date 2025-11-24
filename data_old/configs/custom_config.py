import os
from pathlib import Path

###############################
############ Name #############
###############################

NAME = "zacks_data"  # name and identifier for this configuration

###############################
############ Data #############
###############################

DATA_FOLDER = "/workspace/6s890-finalproject/data/data_files/test" #"/Users/zackduitz/Desktop/organized/MIT/MIT_5th_year_classes/multi-agent_learning/final_project/pgn_files" # folder containing the data files
inter = os.path.join(DATA_FOLDER, NAME)
# H5_FILE = NAME + ".h5"  # H5 file containing data
H5_FILE = "test_chunk.h5" # inter + ".h5"  # H5 file containing data
MAX_MOVE_SEQUENCE_LENGTH = 10  # expected maximum length of move sequences
EXPECTED_ROWS = 12500000  # expected number of rows, approximately, in the data
VAL_SPLIT_FRACTION = 0.9  # marker (% into the data) where the validation split begins
