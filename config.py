from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_PATH = DATASET_DIR / "train.csv"
TEST_PATH = DATASET_DIR / "test.csv"
ERROR_ANALYSIS_PATH = DATASET_DIR / "result/error_analysis.csv"
MODEL_COMPARISON_PATH = DATASET_DIR / "result/model_comparison.csv"

SEED = 42
MAX_LENGTH = 100
EMBEDDING_DIM = 100
LSTM_UNITS = 128
HIDDEN_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SIZE = 0.2
