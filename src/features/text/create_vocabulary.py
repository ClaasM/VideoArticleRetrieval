import os
import pickle
import sys

from gensim import corpora

from src import util

# Tokens dict pickle to read from.
TOKENS_FILE = os.environ["DATA_PATH"] + "/interim/articles/tokens.pickle"
# Vocabulary list pickle to write to.
VOCABULARY_FILE = os.environ["DATA_PATH"] + "/interim/articles/vocabulary.pickle"

articles = pickle.load(open(TOKENS_FILE, "rb"))

"""
dictionary = corpora.Dictionary([tokens for article_id, tokens in articles.items()])
print(len(dictionary))
dictionary.filter_extremes(no_below=5)
print(len(dictionary))
dictionary.filter_extremes(no_below=10)
print(len(dictionary))
dictionary.filter_extremes(no_below=50)
print(len(dictionary))
dictionary.filter_extremes(no_below=100)
print(len(dictionary))
dictionary.filter_extremes(no_below=500)
print(len(dictionary))
dictionary.filter_extremes(no_below=1000)
print(len(dictionary))
"""
"""
dictionary = corpora.Dictionary([tokens for article_id, tokens in articles.items()])
print(len(dictionary))
dictionary.filter_extremes(no_above=0.6)
print(len(dictionary))
dictionary.filter_extremes(no_above=0.5)
print(len(dictionary))
dictionary.filter_extremes(no_above=0.4)
print(len(dictionary))
dictionary.filter_extremes(no_above=0.3)
print(len(dictionary))
dictionary.filter_extremes(no_above=0.2)
print(len(dictionary))
dictionary.filter_extremes(no_above=0.1)
print(len(dictionary))
"""

dictionary = corpora.Dictionary([tokens for article_id, tokens in articles.items()])
# Filter out tokens that could lead to overfitting or have little informativeness and keep 2048 most frequent tokens.
dictionary.filter_extremes(no_above=0.1, no_below=100, keep_n=2048)

pickle.dump(list(dictionary.token2id.keys()), open(VOCABULARY_FILE, "wb+"))
