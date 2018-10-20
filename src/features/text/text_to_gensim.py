"""
Takes the text and saves it as a dictionary and a corpus for gensim to use
"""

import psycopg2
from collections import defaultdict
from gensim import corpora
from src.data.articles import article as article_helper
from src.features.text import tokenize, preprocess


conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
c = conn.cursor()

c.execute("SELECT * FROM articles")
token_frequency = defaultdict(int)
for source_url, in c:
    text = article_helper.load(source_url)
    tokens = tokenize.tokenize(preprocess.preprocess(text))
    for token in tokens:
        token_frequency[token] += 1


# keep words that occur more than once
documents = [[token for token in tweet if token_frequency[token] > 1]
             for tweet in tweets]

# Build a dictionary where for each document each word has its own id
# We stick to the default pruning settings, since they work well.
dictionary = corpora.Dictionary(documents)
dictionary.compactify()
# and save the dictionary for future use
# We use it for the topic model as well as the sentiment model.
dictionary.save('../../data/processed/tweets_sanders.dict')

# Build the corpus: vectors with occurence of each word for each document
# convert tokenized documents to vectors
corpus = [dictionary.doc2bow(doc) for doc in documents]

# and save in Market Matrix format
corpora.MmCorpus.serialize('../../data/processed/tweets_sanders.mm', corpus)
# (This is only used for the LDA topic model)