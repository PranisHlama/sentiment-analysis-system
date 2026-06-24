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
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization


SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

nlp = spacy.load("en_core_web_sm")

import os
for dirname, _, filenames in os.walk('./dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('./dataset/test.csv', encoding = 'latin1')
# print(df.head())
# print("Before removing non null rows:", df.info())

# Drop null rows
df.dropna(inplace=True)

# print(df.sample(20))

# print(df.sample(20))
df["tokenized_text"] = df["text"].astype(str).str.lower().apply(
    lambda x: [token.lemma_ for token in nlp(x) if not token.is_punct] # Use lemma_ for (running -> run) -> Returns base form of words, is_stop for stopwords and is_punct for punctuation(i.e. !!!!!!!!)
)

print(df.head())



# df.to_csv('dataset/train.csv', index=False)
# df2 = pd.read_csv('./dataset/train.csv', encoding = 'latin1')

# print(df2.head(5))