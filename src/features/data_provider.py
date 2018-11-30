"""
TODO documentation
"""

import time
import zlib

import numpy as np
import psycopg2


def preprocess(feature):
    feature = feature - np.min(feature)
    return feature / (np.max(feature) or 1)


class DataProvider:
    def __init__(self, validation_size, test_size):
        conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
        video_cursor = conn.cursor()  # for the videos
        video_cursor.execute(
            "SELECT id, platform, embedding FROM videos WHERE resnet_status='Success' ORDER BY random()")
        # Create the test and validation data used for the ranking. (1,1) mapping for comparability to other research.
        self.ranking_validation_x, self.ranking_validation_y = self.get_1_to_1(validation_size, video_cursor, conn)
        self.ranking_test_x, self.ranking_test_y = self.get_1_to_1(test_size, video_cursor, conn)
        # The data used for training (and used for testing by keras)
        self.train_validation_x, self.train_validation_y = self.get_1_to_n(video_cursor, conn)

    def get_1_to_1(self, size, video_cursor, conn):
        article_cursor = conn.cursor()  # for the articles
        x = np.zeros((size, 2048))
        y = np.zeros((size, 2048))
        for index, (video_id, platform, compressed_video_feature) in zip(range(size), video_cursor):
            video_feature = np.frombuffer(zlib.decompress(compressed_video_feature), np.float32)
            # The video feature is not yet preprocessed, so its preprocessed now (same way as the text) TODO
            y[index] = preprocess(video_feature)

            # get one random article that embeds this video
            article_cursor.execute(
                "SELECT a.embedding FROM article_videos av  "
                "JOIN articles a ON av.source_url = a.source_url "
                "WHERE (av.video_id, av.platform) = (%s,%s) ORDER BY random() LIMIT 1",
                [video_id, platform])
            compressed_article_feature, = article_cursor.fetchone()
            article_feature = np.frombuffer(zlib.decompress(compressed_article_feature), np.float64)
            x[index] = preprocess(article_feature)
        return x, y

    def get_1_to_n(self, video_cursor, conn):
        article_cursor = conn.cursor()  # for the join table
        x = list()
        y = list()
        for video_id, platform, compressed_video_feature in video_cursor:
            video_feature = np.frombuffer(zlib.decompress(compressed_video_feature), np.float32)
            video_feature = preprocess(video_feature)
            # get all articles that embed this video
            article_cursor.execute(
                "SELECT a.embedding FROM article_videos av  "
                "JOIN articles a ON av.source_url = a.source_url "
                "WHERE (av.video_id, av.platform) = (%s,%s)",
                [video_id, platform])
            for compressed_article_feature, in article_cursor:
                # get the features of the articles
                article_feature = np.frombuffer(zlib.decompress(compressed_article_feature), np.float64)
                article_feature = preprocess(article_feature)
                x.append(article_feature)
                y.append(video_feature)

        return np.array(x), np.array(y)
