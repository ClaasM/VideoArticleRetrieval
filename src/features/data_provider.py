"""
TODO documentation
"""

import time
import zlib

import numpy as np
import psycopg2
from sklearn.preprocessing import StandardScaler


class DataProvider:
    def __init__(self, validation_size, test_size):
        conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
        video_cursor = conn.cursor()  # for the videos
        video_cursor.execute("SELECT id, platform, resnet_2048, soundnet_1024 FROM videos "
                             "WHERE resnet_status='Success' AND soundnet_status='Success' ORDER BY random()")

        x, y = self.get_1_to_1_random(video_cursor, conn)

        x = StandardScaler().fit_transform(x, y)

        self.validation_x = np.array(x[:validation_size])
        self.validation_y = np.array(y[:validation_size])

        self.test_x = np.array(x[validation_size:validation_size + test_size])
        self.test_y = np.array(y[validation_size:validation_size + test_size])

        # The data used for training (and used for testing by keras)
        self.train_x = np.array(x[validation_size + test_size:])
        self.train_y = np.array(y[validation_size + test_size:])

        """
        TODO delete if it remains unused
        # Create the test and validation data used for the ranking. (1,1) mapping for comparability to other research.
        self.ranking_validation_x, self.ranking_validation_y = self.get_1_to_1(video_cursor, conn, validation_size)
        self.ranking_test_x, self.ranking_test_y = self.get_1_to_1(video_cursor, conn, test_size)
        # The data used for training (and used for testing by keras)
        self.train_validation_x, self.train_validation_y = self.get_1_to_n(video_cursor, conn)

        self.train_validation_x = StandardScaler().fit_transform(self.train_validation_x)
        self.train_validation_y = StandardScaler().fit_transform(self.train_validation_y)
        """

    def get_1_to_1_random(self, video_cursor, conn, max_size=999999999):
        article_cursor = conn.cursor()  # for the articles
        x = list()
        y = list()
        for index, (video_id, platform, resnet_compressed, soundnet_compressed) in zip(range(max_size), video_cursor):
            # get one random article that embeds this video

            article_cursor.execute(
                "SELECT a.w2v_2048, a.bow_2048 FROM article_videos av  "
                "JOIN articles a ON av.source_url = a.source_url "
                "WHERE (av.video_id, av.platform) = (%s,%s) AND a.text_extraction_status='Success' "
                "ORDER BY random() LIMIT 1",
                [video_id, platform])
            w2v_compressed, bow_compressed = article_cursor.fetchone()
            article_feature = np.concatenate([
                np.frombuffer(zlib.decompress(w2v_compressed), np.float32),
                np.frombuffer(zlib.decompress(bow_compressed), np.float32)
            ])
            video_feature = np.concatenate([
                np.frombuffer(zlib.decompress(resnet_compressed), np.float32),
                np.frombuffer(zlib.decompress(soundnet_compressed), np.float32)
            ])  # np.random.rand(2048)
            x.append(article_feature)
            y.append(video_feature)

        return x, y

    def get_1_to_n(self, video_cursor, conn, max_size=999999999):
        article_cursor = conn.cursor()  # for the join table
        x = list()
        y = list()
        for index, (video_id, platform, compressed_video_feature) in zip(range(max_size), video_cursor):
            video_feature = np.frombuffer(zlib.decompress(compressed_video_feature), np.float32)
            # get all articles that embed this video
            article_cursor.execute(
                "SELECT a.bow_2048 FROM article_videos av  "
                "JOIN articles a ON av.source_url = a.source_url "
                "WHERE (av.video_id, av.platform) = (%s,%s)",
                [video_id, platform])
            for compressed_article_feature, in article_cursor:
                # get the features of the articles
                article_feature = np.frombuffer(zlib.decompress(compressed_article_feature), np.float32)
                x.append(article_feature)
                y.append(video_feature)

        return np.array(x), np.array(y)
