import pandas as pd
import numpy as np
import spacy

import io
import os
import re
import shutil
import string
import tensorflow as tf
import tqdm

from gensim.models import Word2Vec
from tensorflow.keras import Sequential, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
MAX_LENGTH = 100
EMBEDDING_DIM = 100

nlp = spacy.load("en_core_web_sm")

# for dirname, _, filenames in os.walk('./dataset'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

df = pd.read_csv('./dataset/train.csv', encoding = 'latin1')
# print(df.head())
# print("Before removing non null rows:", df.info())

# Drop null rows
df.dropna(inplace=True)

# print(df.sample(20))

# print(df.sample(20))
df["tokenized_text"] = df["text"].astype(str).str.lower().apply(
    lambda x: [token.lemma_ for token in nlp(x) if not token.is_punct] # Use lemma_ for (running -> run) -> Returns base form of words, is_stop for stopwords and is_punct for punctuation(i.e. !!!!!!!!)
)


# Train Word2Vec
sentences = df["tokenized_text"].tolist()

word2vec_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=0 # 0 for Cbow 1 for Skip gram
)

print(word2vec_model.wv["good"])

# Create word-to-index vocabulary
word_index = {}
index = 1

for sentence in df["tokenized_text"]:
    for word in sentence:
        if word not in word_index:
            word_index[word] = index
            index += 1

vocab_size = len(word_index) + 1


# Convert tokens to integer sequences
df["sequence"] = df["tokenized_text"].apply(
    lambda sentence: [word_index[word] for word in sentence if word in word_index]
)

# LSTM needs equal-lengths inputs.(Pad sequence)
X = pad_sequences(
    df["sequence"],
    maxlen = MAX_LENGTH,
    padding = "post",
    truncating = "post"
)

# Prepare sentiment maps
sentiment_map = {
    "positive": 0,
    "negative": 1,
    "neutral": 2
}

y = df["sentiment"].map(sentiment_map).values


# Create Word2Vec embedding matrix
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# build LSTM model
model = Sequential([
    Embedding(
        input_dim = vocab_size,
        output_dim = EMBEDDING_DIM,
        weights = [embedding_matrix],
        input_length = MAX_LENGTH,
        trainable=False
    ),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax") # Because I have positive, negative, neutral
])

# Compile model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics=["accuracy"]
)

# Train model
history = model.fit(
    X,
    y,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)


test_df = pd.read_csv("./dataset/test.csv", encoding="latin1")
test_df.dropna(inplace=True)

test_df["tokenized_text"] = test_df["text"].astype(str).str.lower().apply(
    lambda x: [token.lemma_ for token in nlp(x) if not token.is_punct]
)

test_df["sequence"] = test_df["tokenized_text"].apply(
    lambda sentence: [word_index[word] for word in sentence if word in word_index]
)

X_test = pad_sequences(
    test_df["sequence"],
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

y_test = test_df["sentiment"].map(sentiment_map).values

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

labels = ["positive", "negative", "neutral"]

print(classification_report(y_test, y_pred, target_names=labels))
print(confusion_matrix(y_test, y_pred))

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

history = model.fit(
    X_train,
    y_train,
    epochs = 10,
    batch_size = 32,
    validation_data = (X_val, y_val)
)

print(df.head())
print(df.info())
