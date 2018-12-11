"""
TODO maybe make this a separate package
"""
import time
import random

from keras.callbacks import Callback
import numpy as np
from keras.losses import mean_squared_error
from scipy.spatial import distance
import tensorflow as tf
import keras.backend as K

from src.models.embedding import cosine_proximity


class RankingCallback(Callback):
    def __init__(self, ranking_validation_x, ranking_validation_y):
        super(RankingCallback, self).__init__()
        self.ranking_validation_x = ranking_validation_x
        self.ranking_validation_y = ranking_validation_y

        # Logging stuff.
        log_dir = "/home/claas/logs/%d/" % time.time()
        self.training_writer = tf.summary.FileWriter(log_dir + "/train")
        self.validation_writer = tf.summary.FileWriter(log_dir + "/validation")

    def on_epoch_end(self, epoch, logs=None):
        y_predicted = self.model.predict(self.ranking_validation_x)
        validation_loss = K.eval(K.mean(cosine_proximity(self.ranking_validation_y, y_predicted)))
        ranks = ranking_validation(y_predicted, self.ranking_validation_y)

        # Logging stuff.
        # Training
        self.training_writer.add_summary(
            tf.Summary(value=[
                tf.Summary.Value(tag='loss', simple_value=logs['loss']),
            ]), epoch)
        self.training_writer.flush()

        # Validation
        self.validation_writer.add_summary(
            tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value) for key, value in ranks.items()] + [
                tf.Summary.Value(tag='loss', simple_value=validation_loss)  # validation_loss
            ]), epoch)
        self.validation_writer.flush()


def ranking_validation(y_predicted, y_true):
    # For this model, should be around X
    result = {
        "r1": list(),
        "r5": list(),
        "r10": list(),
        "mean_rank": list(),
        "median_rank": list(),
    }

    for i in range(0, len(y_predicted) - 100, 100):
        # TODO put each step in a separate function
        """
        This is too slow.
        similarities = np.array([
            K.eval(cosine_proximity(np.array([y_predicted[i + j]]), np.array(y_true[i:i + 100]))) for j in range(100)
        ])
        """
        similarities = distance.cdist(y_predicted[i:i + 100], y_true[i:i + 100], 'cosine')  # sqeuclidean
        # Scikit adds 1 to the cosine distance (s.t. 0 is perfect)
        ranks = np.zeros(similarities.shape[0])
        for i in range(similarities.shape[0]):
            # Sort similarities, but keep indices not values
            indices_sorted = np.argsort(similarities[i])
            # The index of i is our rank.
            ranks[i] = np.where(indices_sorted == i)[0][0]

        result["r1"].append(len(np.where(ranks < 1)[0]) / len(ranks))
        result["r5"].append(len(np.where(ranks < 5)[0]) / len(ranks))
        result["r10"].append(len(np.where(ranks < 10)[0]) / len(ranks))
        result["median_rank"].append(np.floor(np.median(ranks)) + 1)
        result["mean_rank"].append(ranks.mean() + 1)

    for key in result:
        result[key] = sum(result[key]) / len(result[key])

    return result
