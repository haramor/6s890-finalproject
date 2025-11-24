# placeholder defaults
DATA_FOLDER = "."
H5_FILE = "output.h5"

# Max number of next moves to encode (input to VAE later)
MAX_MOVE_SEQUENCE_LENGTH = 10

# Overwritten by process_chunks2, but needed to initialize table
EXPECTED_ROWS = 10_000_000

# 90% train / 10% val
VAL_SPLIT_FRACTION = 0.9