import os

# Data paths
DATA_ROOT = "./data"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
VAL_CSV = f"{DATA_ROOT}/val.csv"
TEST_CSV = f"{DATA_ROOT}/test.csv"

# Output path for trained models and logs
OUTPUT_ROOT = "./output"


# Create the data root and output folders if not existing
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# List of files to create under DATA_ROOT
files_to_create = [
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
]

# Create empty files
for filepath in files_to_create:
    # Create file if it does not exist
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            pass  # just create empty file


# Model hyperparameters
MODEL_NAMES = [
    "bert-base-uncased",
]

NUM_LABELS = 4  # Number of classification labels

# Training hyperparameters
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATES = [2e-5, 3e-5]
BATCH_SIZES = [8, 16]
SEED = 42
LOGGING_STEPS = 8

# Tokenizer and model settings
MAX_SEQ_LENGTH = 128
PADDING = True
TRUNCATION = True
