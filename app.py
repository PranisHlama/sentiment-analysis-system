import argparse

MODEL_CHOICES = ("logistic-regression", "lstm", "bert")


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
    runner = get_model_runner(args.model)
    result = runner()
    print(f"Finished {args.model} with accuracy: {result['accuracy']}")


if __name__ == "__main__":
    main()
