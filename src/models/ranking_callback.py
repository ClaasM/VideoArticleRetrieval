"""
TODO maybe make this a separate package
"""
import time

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from scipy.spatial import distance


class RankingCallback(Callback):
    def __init__(self, ranking_validation_x, ranking_validation_y, regularization):
        super(RankingCallback, self).__init__()
        self.ranking_validation_x = ranking_validation_x
        self.ranking_validation_y = ranking_validation_y
        self.regularization = regularization

        # Logging stuff.
        log_dir = "/home/claas/logs/%d/" % time.time()
        self.training_writer = tf.summary.FileWriter(log_dir + "/train")
        self.validation_writer = tf.summary.FileWriter(log_dir + "/validation")

    def on_epoch_end(self, epoch, logs=None):
        y_predicted = self.model.predict(self.ranking_validation_x)
        ranks = ranking_validation(y_predicted, self.ranking_validation_y)
        # Regularization is applied to training loss, so we also need to apply it to validation loss.
        ranks["loss"] += self.get_reg_term()

        # Logging stuff.
        # Training
        self.training_writer.add_summary(
            tf.Summary(value=[
                tf.Summary.Value(tag='loss', simple_value=logs['loss']),
            ]), epoch)
        self.training_writer.flush()

        # Validation
        self.validation_writer.add_summary(
            tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value) for key, value in ranks.items()]), epoch)
        self.validation_writer.flush()

    def get_reg_term(self):
        # Compute regularization term for each layer
        weights = self.model.trainable_weights
        reg_term = 0
        for i, w in enumerate(weights):
            if i % 2 == 0:  # weights from layer i // 2
                w_f = K.flatten(w)
                reg_term += self.regularization * K.sum(K.square(w_f))
        return K.eval(reg_term)


def ranking_validation(y_predicted, y_true):
    # For this model, should be around X
    # TODO dont use lists, use sums
    result = {
        "r1": list(),
        "r5": list(),
        "r10": list(),
        "mean_rank": list(),
        "median_rank": list(),
        "loss": list(),
    }

    for i in range(0, len(y_predicted) - 100, 100):
        # TODO put each step in a separate function
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
        # -1 because that's what keras does.
        result["loss"].append(np.mean([similarities[j][j] for j in range(len(similarities))]) - 1)

    for key in result:
        result[key] = sum(result[key]) / len(result[key])

    return result
