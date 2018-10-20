"""
Takes the text and saves it as a dictionary and a corpus for gensim to use
"""
import os
from collections import defaultdict

import psycopg2
from gensim import corpora

from src.features.text import tokenize

conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
c = conn.cursor()

c.execute("SELECT source_url, text FROM articles WHERE text_extraction_status='Success' LIMIT 1000")

articles = [(source_url, tokenize.tokenize(text)) for (source_url, text) in c]

token_frequency = defaultdict(int)
for source_url, tokens in articles:
    for token in tokens:
        token_frequency[token] += 1

# keep words that occur more than once
documents = [[token for token in tokens if token_frequency[token] > 1]
             for (source_url, tokens) in articles]

# Build a dictionary where for each document each word has its own id
# We stick to the default pruning settings, since they work well.
dictionary = corpora.Dictionary(documents)
dictionary.compactify()
# and save the dictionary for future use
# We use it for the topic model as well as the sentiment model.
dictionary.save(os.environ['MODEL_PATH'] + 'articles.dict')

# Build the corpus: vectors with occurence of each word for each document
# convert tokenized documents to vectors
corpus = [dictionary.doc2bow(doc) for doc in documents]

# and save in Market Matrix format
corpora.MmCorpus.serialize(os.environ['MODEL_PATH'] + 'articles.mm', corpus)
# (This is only used for the LDA topic model)
