import numpy as np
import spacy
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_spacy_model(model_name="en_core_web_sm"):
    return spacy.load(model_name)


def tokenize_texts(texts, nlp, remove_stopwords=True):
    return texts.astype(str).str.lower().apply(
        lambda text: [
            token.lemma_
            for token in nlp(text)
            if not token.is_punct
            and not token.is_space
            and (not remove_stopwords or not token.is_stop)
        ]
    )


def train_word2vec(sentences, embedding_dim, seed):
    return Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=5,
        min_count=1,
        workers=4,
        sg=0,
        seed=seed,
    )


def build_word_index(tokenized_texts):
    word_index = {}

    for sentence in tokenized_texts:
        for word in sentence:
            if word not in word_index:
                word_index[word] = len(word_index) + 1

    return word_index


def texts_to_padded_sequences(tokenized_texts, word_index, max_length):
    sequences = tokenized_texts.apply(
        lambda sentence: [
            word_index[word]
            for word in sentence
            if word in word_index
        ]
    )

    return pad_sequences(
        sequences,
        maxlen=max_length,
        padding="post",
        truncating="post",
    )


def build_embedding_matrix(word_index, word2vec_model, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, index in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[index] = word2vec_model.wv[word]

    return embedding_matrix
