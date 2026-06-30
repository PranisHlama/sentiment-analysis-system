from config import MODEL_COMPARISON_PATH
from src.reporting import save_model_comparison
import argparse

MODEL_CHOICES = ("logistic-regression", "lstm", "bert", "all")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a sentiment analysis model.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="logistic-regression",
        help="Model to train and evaluate.",
    )

    return parser.parse_args()


def get_model_runner(model_name):
    if model_name == "logistic-regression":
        from models.logistic_regression import run_logistic_regression

        return run_logistic_regression

    if model_name == "lstm":
        from models.lstm_classifier import run_lstm

        return run_lstm
    
    if model_name == "bert":
        from models.bert_classifier import run_bert

        return run_bert

    raise ValueError(f"Unsupported model: {model_name}")


def main():
    args = parse_args()

    ### Function calls for each model

    if args.model == "all":
        results = {}

        models = ["logistic-regression", "lstm", "bert"]

        for model in models:
            print(f"\nRunning {model}...")
            runner = get_model_runner(model)
            results[model] = runner()

        save_model_comparison(results, MODEL_COMPARISON_PATH)
        return

    runner = get_model_runner(args.model)
    result = runner()
    save_model_comparison({args.model: result}, MODEL_COMPARISON_PATH)
    print(f"Finished {args.model} with accuracy: {result['accuracy']}")


if __name__ == "__main__":
    main()
