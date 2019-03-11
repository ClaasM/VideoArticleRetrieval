"""
Reads (article_id, [tokens]) from tokens.pickle and writes:
(article_id, w2v)
(article_id, bow)
"""
import json
import sys

import os
import pickle
import psycopg2
from multiprocessing.pool import Pool
import numpy as np
import zlib

# from gensim.models import Word2Vec
from gensim.models import Word2Vec, KeyedVectors

import src

# W2V_FILE = os.environ["MODEL_PATH"] + "/word2vec.model"
from src.visualization.console import StatusVisualization

VOCABULARY_FILE = os.environ["DATA_PATH"] + "/interim/articles/vocabulary.pickle"
W2V_FILE = os.environ["MODEL_PATH"] + "/word2vec.model"

vocabulary = pickle.load(open(VOCABULARY_FILE, "rb"))


def init_worker():
    global model
    model = KeyedVectors.load(W2V_FILE)


# TODO use doc2bow from the dictionary
# TODO divide by max
def count_tokens(tokens):
    token_counter = dict()
    for word in vocabulary:
        token_counter[word] = 0
    for token in tokens:
        if token in token_counter:
            token_counter[token] += 1
    counts = np.array([token_counter[token] for token in vocabulary], dtype=np.float32)
    return zlib.compress(counts, 9)


def w2v_embed(tokens):
    total = np.zeros(2048, dtype=np.float32)
    for token in tokens:
        if token in model:  # Word2Vec model filters some token
            total += model[token]
    return zlib.compress(total / (len(tokens) or 1), 9)


MIN_TOKENS = 50


def extract_features(article):
    article_id, tokens_string = article
    tokens = json.loads(tokens_string)
    if len(tokens) > MIN_TOKENS:
        return "Success", article_id, count_tokens(tokens), w2v_embed(tokens)
    else:
        return "Too few tokens", article_id, None, None


def run():
    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    article_cursor = conn.cursor()
    update_cursor = conn.cursor()
    article_cursor.execute("SELECT count(1) FROM articles WHERE text_extraction_status='Success'")
    article_count, = article_cursor.fetchone()
    # avoid loading all articles into memory.
    article_cursor.execute("SELECT id, tokens FROM articles WHERE text_extraction_status='Success'")

    crawling_progress = StatusVisualization(article_count, update_every=1000)

    with Pool(8, initializer=init_worker) as pool:
        for status, article_id, compressed_bow, compressed_w2v in pool.imap_unordered(extract_features, article_cursor):
            if status == 'Success':
                update_cursor.execute(
                    "UPDATE articles SET bow_2048=%s, w2v_2048=%s, feature_extraction_status='Success' WHERE id=%s",
                    [compressed_bow, compressed_w2v, article_id])
            else:
                update_cursor.execute(
                    "UPDATE articles SET feature_extraction_status=%s WHERE id=%s",
                    [status, article_id])
            crawling_progress.inc()
            conn.commit()


if __name__ == '__main__':
    run()
