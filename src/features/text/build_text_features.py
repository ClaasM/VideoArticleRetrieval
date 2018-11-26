"""
Reads (article_id, [tokens]) from tokens.pickle and writes:
(article_id, w2v)
(article_id, bow)
"""
import os
import pickle
from multiprocessing.pool import Pool

# from gensim.models import Word2Vec
from sklearn import preprocessing
import src

# W2V_FILE = os.environ["MODEL_PATH"] + "/word2vec.model"
TOKENS_FILE = os.environ["DATA_PATH"] + "/interim/articles/tokens.pickle"
VOCABULARY_FILE = os.environ["DATA_PATH"] + "/interim/articles/vocabulary.pickle"

TEXT_FEATURES_FILE = os.environ["DATA_PATH"] + "/interim/articles/features.pickle"

vocabulary = pickle.load(open(VOCABULARY_FILE, "rb"))
articles = pickle.load(open(TOKENS_FILE, "rb"))


def count_tokens(article):
    article_id, tokens = article
    token_counter = dict()
    for word in vocabulary:
        token_counter[word] = 0
    for token in tokens:
        if token in token_counter:
            token_counter[token] += 1
    count_list = [token_counter[token] for token in vocabulary]
    maximum = max(count_list) or 1
    normalized = [float(i) / maximum for i in count_list]
    return article_id, normalized


def run():
    features = dict()
    with Pool(8) as pool:
        for article_id, vector in pool.imap_unordered(count_tokens, articles.items()):
            features[article_id] = vector
            print(article_id)
    pickle.dump(features, open(TEXT_FEATURES_FILE, "wb+"))

    # model = Word2Vec.load(W2V_FILE)
    # print(w2v_model.wv.most_similar(positive="day"))
    # print(w2v_model.wv["day"])


if __name__ == '__main__':
    run()
