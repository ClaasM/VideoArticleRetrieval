import zlib

import psycopg2

import os
import random

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from scipy.spatial import distance

from src.models.w2vv import W2VV_MS

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.utils import generic_utils
# noinspection PyUnresolvedReferences
import tensorboard_logger as tb_logger

import numpy as np

CHECKPOINT_DIR = os.environ["MODEL_PATH"] + "/checkpoints/"

BATCH_SIZE = 100
MAX_EPOCHS = 10000


def calculate_valuation_error(data, predictor):
    videos = np.array([video for video, article in data])
    articles = np.array([article for video, article in data])
    predictions = predictor.model.predict(videos)
    similarities = distance.cdist(predictions, articles, 'cosine')  # TODO is predictions/articles the right way around
    print("Example prediction: %d->%d" % (sum(videos[0]), sum(predictions[0])))
    ranks = np.zeros(similarities.shape[0])
    for i in range(similarities.shape[0]):
        # Sort similarities, but keep indices not values
        indices_sorted = np.argsort(similarities[0])
        # The index of i is our rank.
        ranks[i] = np.where(indices_sorted == i)[0][0]  # TODO maybe np.int64(i)

    # print(ranks)

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return r1, r5, r10, medr, meanr


def run():
    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    cursor = conn.cursor()
    data = list()
    validation_data = list()

    cursor.execute("SELECT id, platform, embedding FROM videos  WHERE resnet_status='Success'")
    videos = cursor.fetchall()
    for video_id, platform, compressed_video_feature in videos:
        video_feature = np.frombuffer(zlib.decompress(compressed_video_feature), np.float32)
        # The video feature is not yet preprocessed, so its preprocessed now (same way as the text)
        video_feature = video_feature - np.min(video_feature)
        video_feature = video_feature / np.max(video_feature)

        # get all articles that embed this video
        cursor.execute("SELECT source_url FROM article_videos  WHERE (video_id, platform) = (%s,%s)",
                       [video_id, platform])
        source_urls = cursor.fetchall()
        articles_features = list()
        # get the features of the articles
        for source_url, in source_urls:
            cursor.execute("SELECT embedding FROM articles  WHERE source_url=%s", [source_url])
            compressed_article_feature, = cursor.fetchone()
            articles_features.append(np.frombuffer(zlib.decompress(compressed_article_feature), np.float64))
        # Add all combinations to the training data.
        for article_feature in articles_features:
            article_feature = article_feature - np.min(article_feature)
            article_feature = article_feature / np.max(article_feature)
            data.append((video_feature, article_feature))
        # For the validation data, we select one article to ensure retrieval-score comparability with other datasets
        validation_article = random.choice(articles_features)
        validation_data.append((video_feature, validation_article))

    print(len(data))
    print(len(validation_data))
    # Just use 100 for now.
    validation_data = validation_data[:100]

    tb_logger.configure(CHECKPOINT_DIR, flush_secs=5)

    # text embedding (text representation)
    n_text_layers = [2048, 2048, 2048]
    # define word2visualvec model
    model = W2VV_MS(n_text_layers)
    # model.plot(model_img_name)
    model.compile_model()
    # model.init_model(opt.init_model_from)
    best_r10 = 0
    decay_count = 0
    learning_rate_count = 0

    step = 0
    batch_count = int(len(data) / BATCH_SIZE)
    # val_img_list: ['1022454332_6af2c1449a', '103106960_e8a41d64f8', '1032122270_ea6f0beedb', '1034...
    # val_sents: ['A child and a woman are at waters edge in a big city .', 'a large lake with...
    for epoch in range(MAX_EPOCHS):
        # print('\nEpoch', epoch)
        tb_logger.log_value('lr', model.get_lr(), step=step)
        progress_bar = generic_utils.Progbar(len(data))
        for batch_index in range(batch_count):
            batch = random.sample(data, BATCH_SIZE)
            batch_videos = [video for video, article in batch]
            batch_articles = [article for video, article in batch]
            # print("Example learning: %.2f->%.2f" % (sum(batch_articles[0]), sum(batch_videos[0])))
            batch_loss = model.model.train_on_batch(np.array(batch_articles), np.array(batch_videos))
            progress_bar.add(BATCH_SIZE, values=[("loss", batch_loss)])

            tb_logger.log_value('loss', batch_loss, step=step)
            tb_logger.log_value('n_step', step, step=step)
            step += 1

        print("\nValidating...")
        r1, r5, r10, medr, meanr = calculate_valuation_error(validation_data, model)
        tb_logger.log_value('r1', r1, step=step)
        tb_logger.log_value('r5', r5, step=step)
        tb_logger.log_value('r10', r10, step=step)
        tb_logger.log_value('median_rank', medr, step=step)
        tb_logger.log_value('mean_rank', meanr, step=step)
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
