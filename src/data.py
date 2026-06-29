import pandas as pd


def load_sentiment_data(path):
    df = pd.read_csv(path, encoding="latin1")
    return df.dropna(subset=["text", "sentiment"]).copy()


def split_features_and_labels(df):
    return df["text"].astype(str), df["sentiment"]
