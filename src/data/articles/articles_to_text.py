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
text_base_path = os.environ["DATA_PATH"] + "/interim/articles_text/"


def extract_text(article):
    article_path, article_file = article


if __name__ == "__main__":
    # Create the text dir if it does not exist yet
    if not os.path.exists(text_base_path):
        os.makedirs(text_base_path)

    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    c = conn.cursor()
    c.execute("SELECT source_url FROM articles")
    articles = [article_helper.get_article_html_filepath(source_url) for (source_url,) in c]
    extractor = BoilerPipeArticleExtractor()
    crawling_progress = CrawlingProgress(len(articles), update_every=100)

    for article_path, article_file in articles:
        article_file_path = os.path.join(article_path, article_file)
        html = util.load_gzip_text(article_file_path)
        text = extractor.get_text(html)
        text_path = os.path.join(text_base_path, os.path.relpath(article_path, articles_base_path))
        if not os.path.exists(text_path):
            os.makedirs(text_path)
        # Just save it under the same name, but in the text directory
        c.execute()
        util.save_gzip_text(os.path.join(text_path, article_file), text)
        crawling_progress.inc(by=1)
