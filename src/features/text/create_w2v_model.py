import os
import pickle

from gensim import corpora
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from src.util import util


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        print(model.get_latest_training_loss())
        self.epoch += 1

    def on_train_begin(self, model):
        print("On train begin")

    def on_train_end(self, model):
        print("On train end")


TOKENS_FILE = os.environ["DATA_PATH"] + "/interim/articles/tokens.pickle"
W2V_FILE = os.environ["MODEL_PATH"] + "/word2vec.model"


def run():
    # Filter out tokens that could lead to overfitting or have little informativeness and keep 2048 most frequent tokens.

    articles = pickle.load(open(TOKENS_FILE, "rb"))
    print("Articles loaded")
    dictionary = corpora.Dictionary([tokens for article_id, tokens in articles.items()])
    print("Documents extracted: %d distinct words" % len(dictionary))
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    print("After filtering: %d distinct words" % len(dictionary))
    model = Word2Vec([tokens for article_id, tokens in articles.items()],
                     size=2048,
                     workers=8,
                     iter=10,
                     compute_loss=True,
                     callbacks=[EpochLogger()])  # , window=5, min_count=1, workers=4
    print("Model trained")
    model.save(W2V_FILE)


if __name__ == '__main__':
    run()

"""
Loss values:
4915771
8681015
11679501
14611220
17304980
19592600
21798460
23879550
25890660
27829164
TODO plot in thesis

"""