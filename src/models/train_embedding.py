import os
import random

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from scipy.spatial import distance

from src.models.text.embeddings import BoW2VecFilterStop, AveWord2VecFilterStop
from src.models.w2vv import W2VV_MS

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.utils import generic_utils
import tensorboard_logger as tb_logger

import numpy as np

CHECKPOINT_DIR = ""  # TODO
VOCABULARY_FILE = os.environ["DATA_PATH"] + "/interim/articles/vocabulary.pickle"
W2V_FILE = os.environ["DATA_PATH"] + ""  # TODO
FEATURE_FILE = os.environ["DATA_PATH"] + ""  # TODO

BATCH_SIZE = 100
MAX_EPOCHS = 100

def run():
    # output visualization script
    runfile_vis = 'do_visual.sh'
    open(runfile_vis, 'w').write('port=$1\ntensorboard --logdir %s --port $port' % CHECKPOINT_DIR)
    os.system('chmod +x %s' % runfile_vis)

    val_per_hist_file = os.path.join(CHECKPOINT_DIR, 'val_per_hist.txt')

    tb_logger.configure(CHECKPOINT_DIR, flush_secs=5)

    # text embedding (text representation)

    bow2vec = BoW2VecFilterStop(VOCABULARY_FILE)
    w2v2vec = AveWord2VecFilterStop(W2V_FILE)
    n_text_layers = [bow2vec.ndims + w2v2vec.ndims, 2048, 2048]
    # define word2visualvec model
    model = W2VV_MS(n_text_layers)

    # model.plot(model_img_name)
    opt = {}

    opt.clipnorm = 5.0
    opt.optimizer = 'rmsprop'
    opt.learning_rate = 0.0001
    model.compile_model('mse', opt=opt)
    # model.init_model(opt.init_model_from)

    # decompressed = np.frombuffer(zlib.decompress(compressed_features), np.float64)
    data = []  # TODO tuples of (video_features_2048(?), [article1_features, ...]) ()

    best_validation_performance = 0
    decay_count = 0
    learning_rate_count = 0

    step = 0
    batch_count = int(len(data) / BATCH_SIZE)
    # val_img_list: ['1022454332_6af2c1449a', '103106960_e8a41d64f8', '1032122270_ea6f0beedb', '1034...
    # val_sents: ['A child and a woman are at waters edge in a big city .', 'a large lake with...
    for epoch in range(MAX_EPOCHS):
        print('\nEpoch', epoch)
        print("Training...")
        print("learning rate: ", model.get_lr())
        tb_logger.log_value('lr', model.get_lr(), step=step)
        progress_bar = generic_utils.Progbar(len(data))
        for batch_index in range(batch_count):
            batch = random.sample(data, BATCH_SIZE)
            batch_videos = [video for video, articles in batch]
            # TODO maybe flatmap before?
            batch_articles = [random.choice(articles) for video, articles in batch]

            batch_loss = model.model.train_on_batch(np.array(batch_articles), np.array(batch_videos))
            progress_bar.add(BATCH_SIZE, values=[("loss", batch_loss)])

            tb_logger.log_value('loss', batch_loss, step=step)
            tb_logger.log_value('n_step', step, step=step)
            step += 1

        print("\nValidating...")
        current_valuation_performance = calculate_valuation_error(data, model)
        tb_logger.log_value('val_accuracy', current_valuation_performance, step=step)
        print('current_valuation_performance: %.3f' % current_valuation_performance)
        learning_rate_count += 1

        if current_valuation_performance > best_validation_performance:
            best_validation_performance = current_valuation_performance
            model.model.save_weights(os.path.join(CHECKPOINT_DIR, 'epoch_%d.h5' % epoch))
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


if __name__ == "__main__":
    run()


def calculate_valuation_error(data, predictor):
    errors = []
    for start in range(0, len(data), BATCH_SIZE):
        end = min(len(data), start + BATCH_SIZE)

        batch = data[start:end]
        batch_videos = [video for video, articles in batch for _ in articles]
        batch_articles = [article for video, articles in batch for article in articles]
        batch_prediction_results = predictor.predict_batch(batch_videos)
        batch_errors = loss(batch_prediction_results, batch_articles)
        errors.extend(batch_errors)

    return np.mean(errors)


# calculate cosine similarity between matrix and matrix
def loss(X, y):
    result = distance.cdist(X, y, 'cosine') - 1
    return result.tolist()
