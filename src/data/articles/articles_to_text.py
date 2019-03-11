import os

import psycopg2

from src import util
from src.data.articles import article as article_helper
from src.data.articles.boilerpipe import BoilerPipeArticleExtractor
from src.visualization.console import StatusVisualization

articles_base_path = os.environ["DATA_PATH"] + "/raw/articles/"

if __name__ == "__main__":
    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    c = conn.cursor()
    c.execute("SELECT source_url FROM articles WHERE text_extraction_status = 'Not Tried'")
    extractor = BoilerPipeArticleExtractor()
    article_urls = list(c)
    crawling_progress = StatusVisualization(len(article_urls), update_every=100)

    for source_url, in article_urls:
        # TODO there should be a method in common/article
        article_path, article_file = article_helper.get_article_html_filepath(source_url)
        html = util.load_gzip_text(os.path.join(article_path, article_file))
        try:
            text = extractor.get_text(html)
            # Save it to the DB
            c.execute("UPDATE articles SET text=%s, text_extraction_status=%s WHERE source_url=%s", [text, "Success", source_url])
            conn.commit()
        except Exception as e:
            # TODO use type(exception).__name__ everywhere
            c.execute("UPDATE articles SET text_extraction_status=%s WHERE source_url=%s", [type(e).__name__, source_url])


        crawling_progress.inc(by=1)
