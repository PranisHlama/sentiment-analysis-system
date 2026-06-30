import argparse
import json
import subprocess
import sys

from config import DATASET_DIR, MODEL_COMPARISON_PATH
from src.reporting import save_model_comparison

MODEL_CHOICES = ("logistic-regression", "lstm", "bert", "all")
RUN_ALL_MODELS = ("logistic-regression", "lstm", "bert")


def result_path_for(model_name):
    safe_name = model_name.replace("-", "_")
    return DATASET_DIR / f"result_{safe_name}.json"


def save_run_result(model_name, result):
    output_path = result_path_for(model_name)
    output_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "accuracy": result["accuracy"],
            },
            indent=2,
        )
    )


def load_run_results(model_names):
    results = {}

    for model_name in model_names:
        with result_path_for(model_name).open() as file:
            results[model_name] = json.load(file)

    return results


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

    if args.model == "all":
        for model_name in RUN_ALL_MODELS:
            print(f"\nRunning {model_name}...")
            subprocess.run(
                [sys.executable, "app.py", "--model", model_name],
                check=True,
            )

        results = load_run_results(RUN_ALL_MODELS)
        save_model_comparison(results, MODEL_COMPARISON_PATH)
        return

    runner = get_model_runner(args.model)
    result = runner()
    save_run_result(args.model, result)
    save_model_comparison({args.model: result}, MODEL_COMPARISON_PATH)
    print(f"Finished {args.model} with accuracy: {result['accuracy']}")


if __name__ == "__main__":
    main()
