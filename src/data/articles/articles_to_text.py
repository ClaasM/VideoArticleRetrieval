import glob
import os

# TODO move boilerpipe
from multiprocessing.pool import Pool

import itertools
from boilerpipe.extract import Extractor
from src import util
from src.visualization.console import CrawlingProgress

articles_base_path = os.environ["DATA_PATH"] + "/raw/articles/"
text_base_path = os.environ["DATA_PATH"] + "/interim/articles_text/"


def extract_text(article):
    #print(article)
    article_path, article_file = article
    article_file_path = os.path.join(article_path, article_file)
    html = util.load_gzip_text(article_file_path)
    text = Extractor(extractor='ArticleExtractor', html=html).getText()
    text_path = os.path.join(text_base_path, os.path.relpath(article_path, articles_base_path))
    if not os.path.exists(text_path):
        os.makedirs(text_path)
    # Just save it under the same name, but in the text directory
    # print(os.path.join(text_path, article_file))
    util.save_gzip_text(os.path.join(text_path, article_file), text)


if __name__ == "__main__":
    # Create the text dir if it does not exist yet
    if not os.path.exists(text_base_path):
        os.makedirs(text_base_path)

    paths = list()


    class Articles:
        def __iter__(self):
            for article_path, _, article_files in os.walk(articles_base_path):
                for article_file in article_files:
                    # We don't want any misc files like .DS_STORE
                    if article_file.endswith(".gzip"):
                        yield article_path, article_file


    #print(len(paths))
    #print(paths[1])


    articles = Articles()

    pool = Pool(1)
    crawling_progress = CrawlingProgress(200000, update_every=100)
    for video in pool.imap_unordered(extract_text, articles):
        crawling_progress.inc(by=1)
    pool.close()
    pool.join()
