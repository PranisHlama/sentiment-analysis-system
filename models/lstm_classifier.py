import json
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import load_model

from config import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    EPOCHS,
    LSTM_ERROR_ANALYSIS_PATH,
    LSTM_MODEL_PATH,
    LSTM_WORD_INDEX_PATH,
    HIDDEN_UNITS,
    LSTM_UNITS,
    MAX_LENGTH,
    SEED,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_SIZE,
)
from src.data import load_sentiment_data
from src.evaluation import print_classification_results, save_error_analysis
from src.labels import ID_TO_LABEL, LABEL_TO_ID, SENTIMENT_LABELS
from src.preprocessing import (
    build_embedding_matrix,
    build_word_index,
    load_spacy_model,
    texts_to_padded_sequences,
    tokenize_texts,
    train_word2vec,
)


def build_lstm_model(vocab_size, embedding_matrix):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_LENGTH,
            trainable=False,
        ),
        LSTM(LSTM_UNITS),
        Dropout(0.5),
        Dense(HIDDEN_UNITS, activation="relu"),
        Dense(len(SENTIMENT_LABELS), activation="softmax"),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


def prepare_lstm_features(train_df, test_df):
    nlp = load_spacy_model()
    train_tokens = tokenize_texts(train_df["text"], nlp)
    test_tokens = tokenize_texts(test_df["text"], nlp)

    word2vec_model = train_word2vec(
        train_tokens.tolist(),
        embedding_dim=EMBEDDING_DIM,
        seed=SEED,
    )
    word_index = build_word_index(train_tokens)
    LSTM_WORD_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LSTM_WORD_INDEX_PATH, "w") as file:
        json.dump(word_index, file)

    embedding_matrix = build_embedding_matrix(
        word_index,
        word2vec_model,
        embedding_dim=EMBEDDING_DIM,
    )

    X = texts_to_padded_sequences(train_tokens, word_index, MAX_LENGTH)
    X_test = texts_to_padded_sequences(test_tokens, word_index, MAX_LENGTH)
    y = train_df["sentiment"].map(LABEL_TO_ID).values
    y_test = test_df["sentiment"].map(LABEL_TO_ID).values
    return X, X_test, y, y_test, embedding_matrix


def prepare_lstm_test_features(test_df):
    with open(LSTM_WORD_INDEX_PATH) as file:
        word_index = json.load(file)

    nlp = load_spacy_model()
    test_tokens = tokenize_texts(test_df["text"], nlp)
    X_test = texts_to_padded_sequences(test_tokens, word_index, MAX_LENGTH)
    y_test = test_df["sentiment"].map(LABEL_TO_ID).values

    return X_test, y_test


def run_lstm(
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    error_analysis_path=LSTM_ERROR_ANALYSIS_PATH,
):
    train_df = load_sentiment_data(train_path)
    test_df = load_sentiment_data(test_path)

    if LSTM_MODEL_PATH.exists() and LSTM_WORD_INDEX_PATH.exists():
        model = load_model(LSTM_MODEL_PATH)
        X_test, y_test = prepare_lstm_test_features(test_df)
        history = None
    else:
        X, X_test, y, y_test, embedding_matrix = prepare_lstm_features(
            train_df,
            test_df,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=VALIDATION_SIZE,
            random_state=SEED,
            stratify=y,
        )

        model = build_lstm_model(
            vocab_size=embedding_matrix.shape[0],
            embedding_matrix=embedding_matrix,
        )
        history = model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
        )

        LSTM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(LSTM_MODEL_PATH)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = print_classification_results(
        y_test,
        y_pred,
        labels=SENTIMENT_LABELS,
    )
    errors = save_error_analysis(
        test_df["text"].astype(str),
        [ID_TO_LABEL[label] for label in y_test],
        [ID_TO_LABEL[label] for label in y_pred],
        error_analysis_path,
    )
    print(errors.head(20))

    return {
        "model": model,
        "history": history,
        "accuracy": accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "y_test": y_test,
        "y_pred": y_pred,
    }
