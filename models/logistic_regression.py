from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd


def build_logistic_regression_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(max_iter=1000))
    ])


def load_data(train_path="./dataset/train.csv", test_path="./dataset/test.csv"):
    train_df = pd.read_csv(train_path, encoding="latin1")
    test_df = pd.read_csv(test_path, encoding="latin1")

    train_df.dropna(subset=["text", "sentiment"], inplace=True)
    test_df.dropna(subset=["text", "sentiment"], inplace=True)

    X_train = train_df["text"].astype(str)
    y_train = train_df["sentiment"]

    X_test = test_df["text"].astype(str)
    y_test = test_df["sentiment"]

    return X_train, X_test, y_train, y_test


def run_logistic_regression(
    train_path="./dataset/train.csv",
    test_path="./dataset/test.csv",
):
    X_train, X_test, y_train, y_test = load_data(train_path, test_path)

    model = build_logistic_regression_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return {
        "model": model,
        "accuracy": accuracy,
        "y_test": y_test,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    run_logistic_regression()
