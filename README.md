# Sentiment Analysis System

## Track 2: Sentiment Analysis System

## Objective

Build a sentiment analysis system that classifies text, such as reviews or social media posts, into positive, negative, or neutral categories.

## Technical Requirements

- Perform text preprocessing:
  - Tokenization
  - Stopword removal
  - Stemming or lemmatization
- Use word embeddings such as Word2Vec or GloVe.
- Implement a sequence model using RNN or LSTM.
- Optional: Compare the deep learning model with a classical machine learning baseline, such as Logistic Regression.
- Optional: Compare with a pretrained transformer model such as BERT or other GenAI architectures.

## Deliverables

- Cleaned dataset and preprocessing pipeline.
- Trained models and comparison.
- Evaluation using accuracy, precision, recall, and F1-score.
- Error analysis section.

## Project Scope

This project focuses on building an end-to-end sentiment classification pipeline. The system should clean raw text data, transform it into useful numerical representations, train sentiment classification models, evaluate model performance, and analyze common prediction errors.

## Expected Model Comparison

The project may include comparisons between:

- A classical baseline model, such as Logistic Regression.
- A deep learning sequence model, such as RNN or LSTM.
- An optional pretrained transformer model, such as BERT.

## Evaluation Metrics

Model performance should be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

## Error Analysis

The final analysis should identify examples where the model makes incorrect predictions and discuss possible causes, such as ambiguous wording, sarcasm, class imbalance, noisy text, or insufficient training data.

Overall accuracy doesn't tell you why the model makes mistakes. Error analysis helps you inspect those mistakes and identify patterns.

For example, if your model frequently predicts Neutral for sentences that are actually Positive, you might discover that:

The training data contains too few positive examples (class imbalance).
Certain positive words or phrases were rare in training.
Sarcasm or context is confusing the model.
Additional preprocessing or a different model could improve performance.

By reviewing the misclassified examples in the generated CSV, you can better understand your model's weaknesses and decide how to improve it. This is why error analysis is a standard step in evaluating machine learning models, especially in tasks like sentiment analysis.