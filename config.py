from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_PATH = DATASET_DIR / "train.csv"
TEST_PATH = DATASET_DIR / "test.csv"

LSTM_ERROR_ANALYSIS_PATH = DATASET_DIR / "error_analysis_lstm.csv"
BERT_ERROR_ANALYSIS_PATH = DATASET_DIR / "error_analysis_bert.csv"
LOGISTIC_REGRESSION_ERROR_ANALYSIS_PATH = DATASET_DIR / "error_analysis_logistic_regression.csv"
MODEL_COMPARISON_PATH = DATASET_DIR / "model_comparison.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"

LOGISTIC_REGRESSION_MODEL_PATH = ARTIFACTS_DIR / "logistic_regression_model.joblib"

LSTM_MODEL_PATH = ARTIFACTS_DIR / "lstm_model.keras"
LSTM_WORD_INDEX_PATH = ARTIFACTS_DIR / "lstm_word_index.json"

BERT_MODEL_DIR = BASE_DIR / "bert_sentiment_model"



SEED = 42
MAX_LENGTH = 100
EMBEDDING_DIM = 100
LSTM_UNITS = 128
HIDDEN_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SIZE = 0.2
