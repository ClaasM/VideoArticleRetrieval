import json

import os
import psycopg2
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
# noinspection PyUnresolvedReferences
import src


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        print("Epoch %d finished with loss: %.2f" % (self.epoch, model.get_latest_training_loss()))
        self.epoch += 1


W2V_FILE = os.environ["MODEL_PATH"] + "/word2vec.model"
KV_FILE = os.environ["MODEL_PATH"] + "/word2vec_kv.model"

"""
TODO move this function to a more appropriate location and refactor other code to use it
"""


def get_article_tokens():
    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    article_cursor = conn.cursor()  # for the videos
    article_cursor.execute(
        "SELECT tokens FROM articles WHERE text_extraction_status='Success' ORDER BY random()")
    return list(map(lambda tokens: json.loads(tokens[0]), article_cursor))


def run():
    articles = get_article_tokens()
    model = Word2Vec(articles,
                     size=2048,
                     workers=8,
                     iter=10,
                     compute_loss=True,
                     callbacks=[EpochLogger()])  # , window=5, min_count=1, workers=4
    model.wv.save(W2V_FILE)


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
