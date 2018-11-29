import os
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from src.features.common import get_data
from src.models.w2vv import W2VV
from src.util.validation import ranking_validation

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.utils import generic_utils

import numpy as np

CHECKPOINT_DIR = os.environ["MODEL_PATH"] + "/checkpoints/"

BATCH_SIZE = 100
MAX_EPOCHS = 10000


def run():
    data, validation_data = get_data()
    print(len(data))
    print(len(validation_data))

    # Just use 100 for now.
    validation_data = validation_data[:100]

    # define word2visualvec model
    model = W2VV()
    model.compile_model()
    best_r10 = 0
    decay_count = 0
    learning_rate_count = 0

    step = 0
    batch_count = int(len(data) / BATCH_SIZE)
    # val_img_list: ['1022454332_6af2c1449a', '103106960_e8a41d64f8', '1032122270_ea6f0beedb', '1034...
    # val_sents: ['A child and a woman are at waters edge in a big city .', 'a large lake with...
    for epoch in range(MAX_EPOCHS):
        # print('\nEpoch', epoch)
        progress_bar = generic_utils.Progbar(len(data))
        for batch_index in range(batch_count):
            batch = random.sample(data, BATCH_SIZE)
            batch_x = [x for x, y in batch]
            batch_y = [y for x, y in batch]
            # index = random.randint(0, len(batch_y) - 1)
            # print("Example learning: %d->%d" % (sum(batch_articles[index]), sum(batch_videos[index])))
            # TODO is this correct?
            batch_loss = model.model.train_on_batch(np.array(batch_x), np.array(batch_y))
            progress_bar.add(BATCH_SIZE, values=[("loss", batch_loss)])
            step += 1

        print("\nValidating...")
        r1, r5, r10, medr, meanr = ranking_validation(validation_data, model.model)
        print((r1, r5, r10, medr, meanr))

        learning_rate_count += 1

        # r10 is the metric we use to choose which model will be used in the end
        if r10 > best_r10:
            best_r10 = r10
            print("Saving model...")
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


if __name__ == "__main__":
    run()

# TODO tb_logger