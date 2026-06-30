# Sentiment Analysis System

## Objective

This project builds an end-to-end sentiment analysis system that classifies text into three categories:

- `negative`
- `neutral`
- `positive`

The system supports a classical machine learning baseline, a deep learning sequence model, and a pretrained transformer model so their performance can be compared on the same dataset.

## Dataset Introduction

The dataset is a tweet-style sentiment classification dataset stored in CSV format under the `dataset/` directory:

- `dataset/train.csv`
- `dataset/test.csv`

Each row represents a short text post and its sentiment label. The main columns used by the models are:

- `text`: the raw text to classify.
- `sentiment`: the target class, one of `negative`, `neutral`, or `positive`.

The CSV files also include metadata such as `textID`, `Time of Tweet`, `Age of User`, `Country`, population, land area, and density. These fields are kept in the dataset, but the current training pipeline uses only the text and sentiment label.

Current label distribution:

| Split | Negative | Neutral | Positive | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 1001 | 1430 | 1103 | 3534 |
| Test | 1001 | 1430 | 1103 | 3534 |

## Project Process

The project follows a standard machine learning workflow:

1. Load the dataset from CSV files.
2. Remove rows with missing `text` or `sentiment` values.
3. Preprocess the text depending on the model type.
4. Train or load the selected model.
5. Predict sentiment labels for the test set.
6. Evaluate the model using classification metrics.
7. Save error analysis output for misclassified examples.
8. Optionally compare all available models in one report.

## Architecture

The project is organized into separate modules for configuration, data loading, preprocessing, model training, evaluation, and reporting.

```text
sentiment-analysis-system/
├── app.py                         # Command-line entry point
├── config.py                      # Paths and training hyperparameters
├── dataset/
│   ├── train.csv                  # Training data
│   ├── test.csv                   # Test data
│   ├── error_analysis.csv         # Generated misclassification report
│   └── model_comparison.csv       # Generated model comparison report
├── models/
│   ├── logistic_regression.py     # TF-IDF + Logistic Regression baseline
│   ├── lstm_classifier.py         # Word2Vec + LSTM classifier
│   └── bert_classifier.py         # BERT transformer classifier
├── src/
│   ├── data.py                    # Dataset loading helpers
│   ├── preprocessing.py           # Tokenization, lemmatization, Word2Vec helpers
│   ├── labels.py                  # Label-to-ID mappings
│   ├── evaluation.py              # Metrics and error analysis
│   └── reporting.py               # Model comparison report generation
└── artifacts/
    └── lstm_word_index.json       # Generated LSTM vocabulary mapping
```

## Model Pipeline

### Logistic Regression Baseline

The Logistic Regression model is implemented as a scikit-learn pipeline:

1. Convert text into TF-IDF features.
2. Use unigrams and bigrams.
3. Remove English stopwords.
4. Train a Logistic Regression classifier.
5. Save or load the model from `artifacts/logistic_regression_model.joblib`.

This baseline is fast to train and gives a useful comparison point for the deeper models.

### LSTM Classifier

The LSTM model is a sequence-based neural network:

1. Load and clean the text.
2. Tokenize and lemmatize text with spaCy.
3. Remove punctuation, spaces, and stopwords.
4. Train Word2Vec embeddings on the training tokens.
5. Build a word index from the training vocabulary.
6. Convert tokenized text into padded integer sequences.
7. Feed sequences into an Embedding layer and LSTM network.
8. Save the trained model to `artifacts/lstm_model.keras`.

The LSTM architecture includes an embedding layer, an LSTM layer, dropout, a dense hidden layer, and a softmax output layer for the three sentiment classes.

### BERT Classifier

The BERT model uses Hugging Face Transformers:

1. Load `bert-base-uncased`.
2. Tokenize text with the BERT tokenizer.
3. Fine-tune `AutoModelForSequenceClassification`.
4. Evaluate on the test dataset.
5. Save the trained model and tokenizer to `bert_sentiment_model/`.

BERT is included as a pretrained transformer comparison against the classical and LSTM approaches.

## Evaluation

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For Logistic Regression and LSTM, misclassified examples are saved to:

```text
dataset/error_analysis.csv
```

When all models are run together, model accuracy results are saved to:

```text
dataset/model_comparison.csv
```

## Error Analysis

Accuracy alone does not explain why a model makes mistakes. The error analysis file records examples where the predicted label does not match the actual label.

Common causes of classification errors include:

- Ambiguous wording
- Sarcasm
- Noisy social media text
- Short or incomplete text
- Class imbalance
- Rare words or phrases not well represented during training

Reviewing these examples helps identify whether improvements should come from better preprocessing, more training data, class balancing, model tuning, or a different model architecture.

## Technical Requirements Covered

- Text preprocessing:
  - Tokenization
  - Stopword removal
  - Lemmatization
- Word embeddings:
  - Word2Vec for the LSTM model
- Sequence model:
  - LSTM
- Classical baseline:
  - TF-IDF + Logistic Regression
- Pretrained transformer:
  - BERT
- Evaluation:
  - Accuracy, precision, recall, F1-score, confusion matrix
- Error analysis:
  - CSV output containing misclassified examples

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The LSTM preprocessing requires the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

Run the Logistic Regression baseline:

```bash
python app.py --model logistic-regression
```

Run the LSTM model:

```bash
python app.py --model lstm
```

Run the BERT model:

```bash
python app.py --model bert
```

Run all models and generate a comparison report:

```bash
python app.py --model all
```

## Outputs

Depending on which model is run, the project may generate:

- `artifacts/logistic_regression_model.joblib`
- `artifacts/lstm_model.keras`
- `artifacts/lstm_word_index.json`
- `bert_sentiment_model/`
- `dataset/error_analysis.csv`
- `dataset/model_comparison.csv`

## Notes

- Existing trained artifacts are reused when available.
- Delete the relevant saved model artifact if you want to retrain a model from scratch.
- BERT training can take significantly longer than Logistic Regression or LSTM, especially without GPU acceleration.
