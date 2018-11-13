"""
Takes the text and saves it as a dictionary and a corpus for gensim to use
"""
import os
import time
import traceback
from collections import defaultdict
from multiprocessing.pool import Pool

import psycopg2
from gensim import corpora

from src.features.text.article_tokenizer import tokenize
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
    return source_url, tokenize(text)


# Parallel tokenization, since it takes by far the most time
crawling_progress = CrawlingProgress(article_count, update_every=1000)
articles = list()
token_count = 0
with Pool(8) as pool:
    for source_url, tokens in pool.imap_unordered(tokenize_parallel, c, chunksize=100):
        articles.append((source_url, tokens))
        token_count += len(tokens)
        crawling_progress.inc()

# articles = [(source_url, tokenize.tokenize(text)) for (source_url, text) in c]
# The rest is not parallelized.
print("Extracting %d tokens took %.2f seconds" % (token_count, time.time() - start))
start = time.time()

token_frequency = defaultdict(int)
for source_url, tokens in articles:
    for token in tokens:
        token_frequency[token] += 1

print("Counting frequencies of %d distinct tokens took %.2f seconds" % (len(token_frequency), time.time() - start))
start = time.time()

# keep words that occur more than once
documents = [[token for token in tokens if token_frequency[token] > 1]
             for (_, tokens) in articles]

print("Filtering down to %d tokens took %.2f seconds"
      % (sum(len(document) for document in documents), time.time() - start))
start = time.time()

# Build a dictionary where for each document each word has its own id
# We stick to the default pruning settings, since they work well.
dictionary = corpora.Dictionary(documents)
dictionary.filter_extremes() # Using Defaults for now
dictionary.compactify()
# and save the dictionary for future use
# We use it for the topic model as well as the sentiment model.
dictionary.save(os.environ['MODEL_PATH'] + 'articles.dict')

print("Compactifying and saving the dictionary took %.2f seconds" % (time.time() - start))
start = time.time()

# Build the corpus: vectors with occurence of each word for each document
# convert tokenized documents to vectors
corpus = [dictionary.doc2bow(doc) for doc in documents]

# and save in Market Matrix format
corpora.MmCorpus.serialize(os.environ['MODEL_PATH'] + 'articles.mm', corpus)
# (This is only used for the LDA topic model)

print("Saving Corpus took %.2f seconds" % (time.time() - start))