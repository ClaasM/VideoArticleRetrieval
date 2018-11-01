"""
Classifies all articles according to the LDA model
"""
import os
from multiprocessing.pool import Pool

import psycopg2
from gensim import corpora
from gensim.models import LdaModel

from src.features.text import tokenize
from src.visualization.console import CrawlingProgress

dictionary = corpora.Dictionary.load(os.environ['MODEL_PATH'] + 'articles.dict')
model = LdaModel.load(os.environ['MODEL_PATH'] + 'articles.lda')


def classify(article):
    source_url, text = article
    tokens = tokenize.tokenize(text)

    """
    doc_bow = [dictionary.doc2bow(token) for token in [tokens]]
    doc_lda = model.get_document_topics(doc_bow,
                                        minimum_probability=None,
                                        minimum_phi_value=None,
                                        per_word_topics=False)
    topics = doc_lda[0]
    topics_ret = dict()
    for topic in topics:
        print(topics)
        topic_id = topic[0]
        topic_probability = topic[1]
        """
    # TODO this ignores topics with <1% probability
    topics = model[dictionary.doc2bow(tokens)]
    return source_url, topics


def run():
    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    c = conn.cursor()
    # Reset the topics table
    c.execute("DROP TABLE IF EXISTS topics")
    num_topics = 10
    query = "CREATE TABLE topics (source_url TEXT PRIMARY KEY, "
    query += ",".join("topic_%d FLOAT DEFAULT 0" % index for index in range(0, num_topics))
    query += ")"
    c.execute(query)
    conn.commit()
    c.execute("SELECT count(1) FROM articles WHERE text_extraction_status='Success'")
    article_count, = c.fetchone()

    crawling_progress = CrawlingProgress(article_count, update_every=1000)
    articles_cursor = conn.cursor()
    articles_cursor.execute("SELECT source_url, text FROM articles WHERE text_extraction_status='Success'")
    # parallel classification
    with Pool(8) as pool:  # 16 seems to be around optimum
        for source_url, topics in pool.imap_unordered(classify, articles_cursor, chunksize=100):
            query = "INSERT INTO topics (source_url, " + ",".join("topic_%d" % topic[0] for topic in topics) + ")" \
                    + ("VALUES ('%s', " % source_url) + ",".join("%f" % topic[1] for topic in topics) + ")"
            c.execute(query)
            conn.commit()
            crawling_progress.inc(1)


if __name__ == "__main__":
    run()
