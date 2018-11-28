"""
Reads (article_id, [tokens]) from tokens.pickle and writes:
(article_id, w2v)
(article_id, bow)
"""
import json
import os
import pickle
import psycopg2
from multiprocessing.pool import Pool
import numpy as np
import zlib

# from gensim.models import Word2Vec
import src

# W2V_FILE = os.environ["MODEL_PATH"] + "/word2vec.model"
from src.visualization.console import CrawlingProgress

VOCABULARY_FILE = os.environ["DATA_PATH"] + "/interim/articles/vocabulary.pickle"

vocabulary = pickle.load(open(VOCABULARY_FILE, "rb"))


def count_tokens(article):
    article_id, tokens_string = article
    tokens = json.loads(tokens_string)
    token_counter = dict()
    for word in vocabulary:
        token_counter[word] = 0
    for token in tokens:
        if token in token_counter:
            token_counter[token] += 1
    count_list = [token_counter[token] for token in vocabulary]
    maximum = max(count_list) or 1
    normalized = [float(i) / maximum for i in count_list]
    return article_id, zlib.compress(np.array(normalized), 9)


def run():
    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    article_cursor = conn.cursor()
    update_cursor = conn.cursor()
    article_cursor.execute("SELECT count(1) FROM articles WHERE text_extraction_status='Success'")
    article_count, = article_cursor.fetchone()
    # avoid loading all articles into memory.
    article_cursor.execute("SELECT id, tokens FROM articles WHERE text_extraction_status='Success'")

    crawling_progress = CrawlingProgress(article_count, update_every=1000)

    with Pool(8) as pool:
        for article_id, compressed_features in pool.imap_unordered(count_tokens, article_cursor):
            update_cursor.execute("UPDATE articles SET embedding=%s WHERE id=%s", [compressed_features, article_id])
            crawling_progress.inc()
        conn.commit()

    # model = Word2Vec.load(W2V_FILE)
    # print(w2v_model.wv.most_similar(positive="day"))
    # print(w2v_model.wv["day"])


if __name__ == '__main__':
    run()
