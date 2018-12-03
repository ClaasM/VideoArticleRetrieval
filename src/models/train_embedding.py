import pickle
import time

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard

from src.features.data_provider import DataProvider
from src.models.embedding import build_model
from src.models.ranking_callback import RankingCallback

# TODO remove this (its just here so that feature extraction can run simultaneously)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

CHECKPOINT_DIR = os.environ["MODEL_PATH"] + "/checkpoints/"

BATCH_SIZE = 100
EPOCHS = 100

# Smaller values make for faster valuation.
TEST_SIZE = 1000
VALIDATION_SIZE = 1000

DATA_CACHE = os.environ['MODEL_PATH'] + "/data_provider.pickle"


def run():
    # During parameter tuning, this makes things a bit faster:
    #data_provider = DataProvider(VALIDATION_SIZE, TEST_SIZE)
    #pickle.dump(data_provider, open(DATA_CACHE, "wb+"))
    data_provider = pickle.load(open(DATA_CACHE, "rb"))

    # tensorboard_logger = TensorBoard(log_dir="/home/claas/logs/%d" % time.time())
    ranking_callback = RankingCallback(data_provider.validation_x,
                                       data_provider.validation_y, )
    # define model
    model = build_model(
        data_provider.train_x.shape[1],
        data_provider.train_y.shape[1]
    )

    print("Done building model")
    model.fit(x=data_provider.train_x,
              y=data_provider.train_y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=[ranking_callback],  # ranking_callback
              validation_data=(data_provider.validation_x, data_provider.validation_y))

    # TODO validate model on best epoch with data_provider.ranking_test_x
    # print(test_result)


if __name__ == "__main__":
    run()
