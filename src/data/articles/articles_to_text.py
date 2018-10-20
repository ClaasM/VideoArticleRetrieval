import glob
import os

# TODO move boilerpipe
from multiprocessing.pool import Pool

import jpype

from src.data.articles import article as article_helper

import itertools

import psycopg2
from src import util
from src.data.articles.boilerpipe import BoilerPipeArticleExtractor
from src.visualization.console import CrawlingProgress

articles_base_path = os.environ["DATA_PATH"] + "/raw/articles/"

if __name__ == "__main__":
    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    c = conn.cursor()
    c.execute("SELECT source_url FROM articles WHERE text_extraction_status = 'Not Tried'")
    extractor = BoilerPipeArticleExtractor()
    article_urls = list(c)
    crawling_progress = CrawlingProgress(len(article_urls), update_every=100)

    for source_url, in article_urls:
        article_path, article_file = article_helper.get_article_html_filepath(source_url)

        article_file_path = os.path.join(article_path, article_file)
        html = util.load_gzip_text(article_file_path)
        try:
            text = extractor.get_text(html)
            # Save it to the DB
            c.execute("UPDATE articles SET text=%s, text_extraction_status=%s WHERE source_url=%s", [text, "Success", source_url])
            conn.commit()
        except Exception as e:
            # TODO use type(exception).__name__ everywhere
            c.execute("UPDATE articles SET text_extraction_status=%s WHERE source_url=%s", [type(e).__name__, source_url])


        crawling_progress.inc(by=1)
