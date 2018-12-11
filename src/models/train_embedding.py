import csv
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import src
import zlib

import os
import numpy as np

BATCH_SIZE = 100
EPOCHS = 100

# Smaller values make for faster valuation.
TEST_SIZE = 1000
VALIDATION_SIZE = 1000

DATA_CACHE = os.environ['MODEL_PATH'] + "/data_provider.pickle"
PICKLE_FILE = os.environ["DATA_PATH"] + "/interim/data.pickle"


def get_data():
    data = pickle.load(open(PICKLE_FILE, "rb"))
    x = list()
    y = list()
    for w2v_2048, bow_2048, resnet_2048, soundnet_1024, i3d_rgb_1024 in data:
        x.append(np.concatenate([
            w2v_2048,
            # np.zeros(2048),
            bow_2048
        ]))
        y.append(np.concatenate([
            resnet_2048,
            # soundnet_1024,
            # i3d_rgb_1024,
        ]))

    x, y = shuffle(x, y)
    x = StandardScaler().fit_transform(x, y)

    data_split = dict()
    data_split["validation_x"] = np.array(x[:VALIDATION_SIZE])
    data_split["validation_y"] = np.array(y[:VALIDATION_SIZE])
    data_split["test_x"] = np.array(x[VALIDATION_SIZE:VALIDATION_SIZE + TEST_SIZE])
    data_split["test_y"] = np.array(y[VALIDATION_SIZE:VALIDATION_SIZE + TEST_SIZE])
    data_split["train_x"] = np.array(x[VALIDATION_SIZE + TEST_SIZE:])
    data_split["train_y"] = np.array(y[VALIDATION_SIZE + TEST_SIZE:])
    return data_split


def train():
    data = get_data()

    from src.models.embedding import build_model
    from src.models.ranking_callback import RankingCallback

    ranking_callback = RankingCallback(data["validation_x"],
                                       data["validation_y"])

    # define model
    model = build_model(
        data["train_x"].shape[1],
        data["train_y"].shape[1]
    )

    model.fit(x=data["train_x"],
              y=data["train_y"],
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=[ranking_callback],
              validation_data=(data["validation_x"], data["validation_y"]))

    # TODO validate model on best epoch with data_provider.ranking_test_x


if __name__ == "__main__":
    train()
    # get_data()
