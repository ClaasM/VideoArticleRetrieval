"""
Takes the text and saves it as an array of tokens.
Useful helper, during modeling, because extracting the tokes is comparatively slow.
"""
import os
import pickle
from multiprocessing.pool import Pool

import psycopg2

from src.features.text.article_tokenizer import tokenize
from src.visualization.console import CrawlingProgress

TOKENS_FILE = os.environ["DATA_PATH"] + "/interim/articles/tokens.pickle"

conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
c = conn.cursor()
# Get the count.
c.execute("SELECT count(1) FROM articles WHERE text_extraction_status='Success'")
article_count, = c.fetchone()
# avoid loading all articles into memory.
c.execute("SELECT id, text FROM articles WHERE text_extraction_status='Success'")


def tokenize_parallel(article):
    source_url, text = article
    return source_url, tokenize(text)


# Parallel tokenization, since it takes by far the most time
articles = dict()
tokens_count = 0
crawling_progress = CrawlingProgress(article_count, update_every=1000)

with Pool(8) as pool:
    for article_id, tokens in pool.imap_unordered(tokenize_parallel, c, chunksize=100):
        tokens_count += len(tokens)
        crawling_progress.inc()
        articles[article_id] = tokens

pickle.dump(articles, open(TOKENS_FILE, "wb+"))
print("Extracted %d tokens." % tokens_count)