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

    ### Run each model in a seperate process

    if args.model == "all":
        import subprocess
        import sys

        models = ["logistic-regression", "lstm", "bert"]

        for model in models:
            print(f"\nRunning {model}...")
            subprocess.run(
                [sys.executable, "app.py", "--model", model],
                check=True,
            )

        return

    runner = get_model_runner(args.model)
    result = runner()
    print(f"Finished {args.model} with accuracy: {result['accuracy']}")


if __name__ == "__main__":
    main()
