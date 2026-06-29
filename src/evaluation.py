import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def print_classification_results(y_true, y_pred, labels=None):
    accuracy = accuracy_score(y_true, y_pred)

    print("Accuracy:", accuracy)
    print(classification_report(y_true, y_pred, target_names=labels))
    print(confusion_matrix(y_true, y_pred))

    return accuracy


def save_error_analysis(texts, actual, predicted, output_path):
    errors = pd.DataFrame({
        "text": list(texts),
        "actual": list(actual),
        "predicted": list(predicted),
    })
    errors = errors[errors["actual"] != errors["predicted"]] # Filter out the model's mistakes
    errors.to_csv(output_path, index=False)

    return errors
