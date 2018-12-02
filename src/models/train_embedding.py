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


def run():

    data_provider = DataProvider(VALIDATION_SIZE, TEST_SIZE)
    # tensorboard_logger = TensorBoard(log_dir="/home/claas/logs/%d" % time.time())
    ranking_callback = RankingCallback(data_provider.ranking_validation_x,
                                       data_provider.ranking_validation_y,)
    # define word2visualvec model
    model = build_model()

    print("Done building model")
    model.fit(x=data_provider.train_validation_x,
              y=data_provider.train_validation_y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=[ranking_callback], # ranking_callback
              validation_split=0.1)

    # TODO validate model on best epoch with data_provider.ranking_test_x
    # print(test_result)


if __name__ == "__main__":
    run()