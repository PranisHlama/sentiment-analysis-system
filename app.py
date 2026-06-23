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

from tensorflow.keras import Sequential, layers
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences


SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
MAX_LENGTH = 100


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

print(df.head())
