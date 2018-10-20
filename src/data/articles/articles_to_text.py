import glob
import os

# TODO move boilerpipe
from multiprocessing.pool import Pool
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
    c.execute("SELECT source_url FROM articles")
    extractor = BoilerPipeArticleExtractor()
    article_urls = list(c)
    crawling_progress = CrawlingProgress(len(article_urls), update_every=100)

    for source_url, in article_urls:
        article_path, article_file = article_helper.get_article_html_filepath(source_url)

        article_file_path = os.path.join(article_path, article_file)
        html = util.load_gzip_text(article_file_path)
        text = extractor.get_text(html)
        # Save it to the DB
        c.execute("UPDATE articles SET text=%s WHERE source_url=%s", [text, source_url])
        conn.commit()
        crawling_progress.inc(by=1)
