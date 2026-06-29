from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import ERROR_ANALYSIS_PATH, TEST_PATH, TRAIN_PATH
from src.data import load_sentiment_data, split_features_and_labels
from src.evaluation import print_classification_results, save_error_analysis


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


def run_logistic_regression(
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    error_analysis_path=ERROR_ANALYSIS_PATH,
):
    train_df = load_sentiment_data(train_path)
    test_df = load_sentiment_data(test_path)
    X_train, y_train = split_features_and_labels(train_df)
    X_test, y_test = split_features_and_labels(test_df)

    model = build_logistic_regression_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = print_classification_results(y_test, y_pred)
    errors = save_error_analysis(X_test, y_test, y_pred, error_analysis_path)
    print(errors.head(20))

    return {
        "model": model,
        "accuracy": accuracy,
        "y_test": y_test,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    run_logistic_regression()
