"""
Takes the text and saves it as an array of tokens.
Useful helper, during modeling, because extracting the tokes is comparatively slow.
"""
import json
import os
from multiprocessing.pool import Pool

import psycopg2

from src.features.text.article_tokenizer import tokenize
from src.visualization.console import StatusVisualization

def tokenize_parallel(article):
    source_url, text = article
    return source_url, json.dumps(tokenize(text))


def run():
    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    article_cursor = conn.cursor()
    update_cursor = conn.cursor()
    # Get the count.
    article_cursor.execute("SELECT count(1) FROM articles WHERE text_extraction_status='Success'")
    article_count, = article_cursor.fetchone()
    # avoid loading all articles into memory.
    article_cursor.execute("SELECT id, text FROM articles WHERE text_extraction_status='Success'")
    # Parallel tokenization, since it takes by far the most time

    crawling_progress = StatusVisualization(article_count, update_every=1000)
    with Pool(8) as pool:
        for article_id, tokens_string in pool.imap_unordered(tokenize_parallel, article_cursor, chunksize=100):
            update_cursor.execute("UPDATE articles SET tokens=%s WHERE id=%s", [tokens_string, article_id])
            crawling_progress.inc()

        conn.commit()


if __name__ == '__main__':
    run()
