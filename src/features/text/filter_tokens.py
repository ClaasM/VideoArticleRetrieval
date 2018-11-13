"""
Takes the text and saves it as an array of tokens
"""
import pickle
from multiprocessing.pool import Pool

import psycopg2

from src.features.text.article_tokenizer import tokenize

conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
c = conn.cursor()
# avoid loading all articles into memory.
c.execute("SELECT source_url, text FROM articles WHERE text_extraction_status='Success' ORDER BY RANDOM()")


def tokenize_parallel(article):
    source_url, text = article
    return source_url, tokenize(text)


# Parallel tokenization, since it takes by far the most time
articles = list()
with Pool(8) as pool:
    for source_url, tokens in pool.imap_unordered(tokenize_parallel, c, chunksize=100):
        articles.append(tokens)

pickle.dump(articles, open("tokens", "wb+"))