"""
TODO maybe make this a separate package
"""

from keras.callbacks import Callback
import numpy as np
from scipy.spatial import distance


class RankingCallback(Callback):
    def __init__(self, ranking_validation_x, ranking_validation_y):
        super(RankingCallback, self).__init__()
        self.ranking_validation_x = ranking_validation_x
        self.ranking_validation_y = ranking_validation_y

    def on_epoch_end(self, epoch, logs=None):
        y_predicted = self.model.predict(self.ranking_validation_x)
        ranks = ranking_validation(y_predicted, self.ranking_validation_y)
        # print(self.model)
        # print(len(self.ranking_validation_x))
        # TODO use the model to get the metrics from ranking test data
        print(ranks)


def ranking_validation(y_predicted, y_true):
    # TODO is predictions/articles the right way around
    similarities = distance.cdist(y_predicted, y_true, 'cosine')
    ranks = np.zeros(similarities.shape[0])
    for i in range(similarities.shape[0]):
        # Sort similarities, but keep indices not values
        indices_sorted = np.argsort(similarities[i])
        # The index of i is our rank.
        ranks[i] = np.where(indices_sorted == i)[0][0]  # TODO maybe np.int64(i)

    result = dict()
    result["r1"] = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    result["r5"] = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    result["r10"] = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    result["median_rank"] = np.floor(np.median(ranks)) + 1
    result["mean_rank"] = ranks.mean() + 1
    return result
