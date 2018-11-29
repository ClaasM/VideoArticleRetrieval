import zlib

import psycopg2
import numpy as np
import random


def preprocess(feature):
    features = feature - np.min(feature)
    return feature / (np.max(feature) or 1)


def get_data():
    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    cursor = conn.cursor()
    data = list()
    validation_data = list()

    cursor.execute("SELECT id, platform, embedding FROM videos  WHERE resnet_status='Success'")
    videos = cursor.fetchall()
    for video_id, platform, compressed_video_feature in videos:
        video_feature = np.frombuffer(zlib.decompress(compressed_video_feature), np.float32)
        # The video feature is not yet preprocessed, so its preprocessed now (same way as the text)
        video_feature = preprocess(video_feature)

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
            article_feature = preprocess(article_feature)
            data.append((video_feature, article_feature))
        # For the validation data, we select one article to ensure retrieval-score comparability with other datasets
        validation_article = random.choice(articles_features)
        validation_data.append((video_feature, validation_article))

    return data, validation_data
