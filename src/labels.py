SENTIMENT_LABELS = ["negative", "neutral", "positive"]

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
