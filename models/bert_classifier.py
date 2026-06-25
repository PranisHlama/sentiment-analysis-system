import numpy as np
import pandas as pd
import evaluate

from datasets import Dataset
from transformers import (
      AutoTokenizer,
      AutoModelForSequenceClassification,
      DataCollatorWithPadding,
      TrainingArguments,
      Trainer,
)

from transformers import pipeline


MODEL_NAME = "bert-base-uncased"

label2id = {
    "positive": 0,
    "negative": 1,
    "neutral": 2,
}

id2label = {
    0: "positive",
    1: "negative",
    2: "neutral",
}


def load_dataset(path):
    df = pd.read_csv(path, encoding="latin1")
    df = df.dropna(subset=["text", "sentiment"])

    df = df[["text", "sentiment"]].copy()
    df["label"] = df["sentiment"].map(label2id)

    return Dataset.from_pandas(df[["text", "label"]])


train_dataset = load_dataset("../dataset/train.csv")
test_dataset = load_dataset("../dataset/test.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # Autotokenizer converts tweets into BERT Tokens


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)


train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained( # Adds a classification head on top of BERT.
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./bert_sentiment_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
) # Trainer fine-tunes BERT on positive, negative and neutral labels.

trainer.train()
trainer.evaluate()

trainer.save_model("./bert_sentiment_model")
tokenizer.save_pretrained("./bert_sentiment_model")


classifier = pipeline(
    "sentiment-analysis",
    model="./bert_sentiment_model",
    tokenizer="./bert_sentiment_model",
)

print(classifier("I love this product!"))