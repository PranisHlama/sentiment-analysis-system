import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from config import TRAIN_PATH, TEST_PATH, BERT_MODEL_DIR, BERT_ERROR_ANALYSIS_PATH
from src.labels import LABEL_TO_ID, ID_TO_LABEL
from src.evaluation import save_error_analysis

MODEL_NAME = "bert-base-uncased"


def load_dataset(path):
    df = pd.read_csv(path, encoding="latin1")
    df = df.dropna(subset=["text", "sentiment"])

    df = df[["text", "sentiment"]].copy()
    df["label"] = df["sentiment"].map(LABEL_TO_ID)

    return Dataset.from_pandas(df[["text", "label"]])



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


def run_bert(train_path=TRAIN_PATH, test_path=TEST_PATH, error_analysis_path=BERT_ERROR_ANALYSIS_PATH):
    train_dataset = load_dataset(train_path)
    test_dataset = load_dataset(test_path)

    if BERT_MODEL_DIR.exists():
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL_TO_ID),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )

    train_dataset = train_dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=128),
        batched=True,
    )
    test_dataset = test_dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=128),
        batched=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(BERT_MODEL_DIR),
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
    )

    if not BERT_MODEL_DIR.exists():
        trainer.train()
        trainer.save_model(BERT_MODEL_DIR)
        tokenizer.save_pretrained(BERT_MODEL_DIR)

    predictions = trainer.predict(test_dataset)
    metrics = predictions.metrics
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_test = np.array(test_dataset["label"])

    errors = save_error_analysis(
        test_dataset["text"],
        [ID_TO_LABEL[label] for label in y_test],
        [ID_TO_LABEL[label] for label in y_pred],
        error_analysis_path,
    )

    print(errors.head(20))

    return {
        "model": model,
        "accuracy": metrics.get("test_accuracy"),
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
    }

if __name__ == "__main__":
    run_bert()
