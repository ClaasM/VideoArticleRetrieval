import os
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from src.features.common import get_data
from src.models.w2vv import W2VV
from src.util.validation import ranking_validation

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

from keras.utils import generic_utils

import numpy as np

CHECKPOINT_DIR = os.environ["MODEL_PATH"] + "/checkpoints/"

BATCH_SIZE = 100
MAX_EPOCHS = 100


def run():
    data, validation_data = get_data()
    print(len(data))
    print(len(validation_data))

    # Just use 100 for now.
    validation_data = validation_data[:100]

    # define word2visualvec model
    model = W2VV()
    model.compile_model()
    best_result = {"mean_rank": len(validation_data)} # worst case
    decay_count = 0
    learning_rate_count = 0

    step = 0
    batch_count = int(len(data) / BATCH_SIZE)
    for epoch in range(MAX_EPOCHS):
        print('\nEpoch', epoch)
        progress_bar = generic_utils.Progbar(len(data))
        for batch_index in range(batch_count):
            batch = random.sample(data, BATCH_SIZE)
            batch_x = [x for x, y in batch]
            batch_y = [y for x, y in batch]
            batch_loss = model.model.train_on_batch(np.array(batch_x), np.array(batch_y))
            progress_bar.add(BATCH_SIZE, values=[("loss", batch_loss)])
            step += 1

        print("\nValidating...")
        result = ranking_validation(validation_data, model.model)

        learning_rate_count += 1

        # r10 is the metric we use to choose which model will be used in the end
        if result["mean_rank"] < best_result["mean_rank"]:
            best_result = result
            print(best_result)
            # TODO
            # model.model.save_weights(os.path.join(CHECKPOINT_DIR, 'epoch_%d.h5' % epoch))
        """
        elif learning_rate_count >= 3:
            # when the validation performance has decreased after an epoch,
            # we divide the learning rate by 2 and continue training;
            # but we use each learning rate for at least 3 epochs.
            print("Decaying learning rate...")
            model.decay_lr(0.5)
            learning_rate_count = 0
            # Dont decay LR more than 10 times.
            decay_count += 1
            if decay_count > 10:
                print("Early stopping happened")
                break
        """
    print(best_result)


if __name__ == "__main__":
    run()

# TODO tb_logger

"""
Results (best mean_rank):
mape, euclidean: {'median_rank': 42.0, 'mean_rank': 45.36, 'r5': 10.0, 'r10': 15.0, 'r1': 3.0}
mse, cosine: {'r10': 87.0, 'r5': 79.0, 'median_rank': 1.0, 'mean_rank': 6.37, 'r1': 58.0}

"""