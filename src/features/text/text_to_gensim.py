"""
Takes the text and saves it as a dictionary and a corpus for gensim to use
"""
import os
from collections import defaultdict
from multiprocessing.pool import Pool

import psycopg2
from gensim import corpora
from src.features.text import tokenize

import time

from src.visualization.console import CrawlingProgress

start = time.time()
conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
c = conn.cursor()
c.execute("SELECT count(1) FROM articles WHERE text_extraction_status='Success'")
article_count, = c.fetchone()
# avoid loading all articles into memory.
c.execute("SELECT source_url, text FROM articles WHERE text_extraction_status='Success'")


def tokenize_parallel(article):
    source_url, text = article
    return source_url, tokenize.tokenize(text)


# Parallel tokenization, since it takes by far the most time
crawling_progress = CrawlingProgress(article_count, update_every=1000)
with Pool(8) as pool:
    articles = list()
    for article in pool.imap_unordered(tokenize_parallel, c, chunksize=100):
        articles.append(article)
        crawling_progress.inc()

# articles = [(source_url, tokenize.tokenize(text)) for (source_url, text) in c]
# The rest is not parallelized.
print("Querying + Tokenization took %d seconds" % (time.time() - start))
start = time.time()

token_frequency = defaultdict(int)
for source_url, tokens in articles:
    for token in tokens:
        token_frequency[token] += 1

print("Frequency counting took %d seconds" % (time.time() - start))
start = time.time()

# keep words that occur more than once
documents = [[token for token in tokens if token_frequency[token] > 1]
             for (_, tokens) in articles]

print("Filtering words took %d seconds" % (time.time() - start))
start = time.time()

# Build a dictionary where for each document each word has its own id
# We stick to the default pruning settings, since they work well.
dictionary = corpora.Dictionary(documents)
dictionary.compactify()
# and save the dictionary for future use
# We use it for the topic model as well as the sentiment model.
dictionary.save(os.environ['MODEL_PATH'] + 'articles.dict')

print("Saving Dictionary took %d seconds" % (time.time() - start))
start = time.time()

# Build the corpus: vectors with occurence of each word for each document
# convert tokenized documents to vectors
corpus = [dictionary.doc2bow(doc) for doc in documents]

# and save in Market Matrix format
corpora.MmCorpus.serialize(os.environ['MODEL_PATH'] + 'articles.mm', corpus)
# (This is only used for the LDA topic model)

print("Saving Corpus took %d seconds" % (time.time() - start))
