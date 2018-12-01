"""
TODO maybe make this a separate package
"""
import time

import random

from keras.callbacks import Callback
import numpy as np
from scipy.spatial import distance
import tensorflow as tf
import keras.backend as K


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
                tf.Summary.Value(tag='loss', simple_value=logs['val_loss'])
            ]), epoch)
        self.validation_writer.flush()


def ranking_validation(y_predicted, y_true):
    similarities = distance.cdist(y_predicted, y_true, 'cosine')
    ranks = np.zeros(similarities.shape[0])
    for i in range(similarities.shape[0]):
        # Sort similarities, but keep indices not values
        indices_sorted = np.argsort(similarities[i])
        # The index of i is our rank.
        ranks[i] = np.where(indices_sorted == i)[0][0]

    result = dict()
    result["r1"] = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    result["r5"] = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    result["r10"] = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    result["median_rank"] = np.floor(np.median(ranks)) + 1
    result["mean_rank"] = ranks.mean() + 1
    return result
