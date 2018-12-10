"""
Retrieves the data from the database and saves it in a csv for more convenient access during modeling.
"""
import pickle

import psycopg2
import os
import src
import numpy as np
import zlib

PICKLE_FILE = os.environ["DATA_PATH"] + "/interim/data.pickle"


def run():
    data = list()

    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    video_cursor = conn.cursor()  # for the videos
    video_cursor.execute("""
        SELECT id, platform, resnet_2048, soundnet_1024, i3d_rgb_1024
        FROM videos
        WHERE resnet_status = 'Success'
          AND soundnet_status = 'Success'
          AND i3d_rgb_status = 'Success'
        ORDER BY random() 
    """)

    article_cursor = conn.cursor()  # for the articles
    for video_id, platform, resnet_2048, soundnet_1024, i3d_rgb_1024 in video_cursor:
        # get one random article that embeds this video
        article_cursor.execute("""
            SELECT a.w2v_2048, a.bow_2048
            FROM article_videos av
                   JOIN articles a ON av.source_url = a.source_url
            WHERE (av.video_id, av.platform) = (%s,%s)
              AND a.text_extraction_status = 'Success'
            ORDER BY random()
            LIMIT 1 
        """, [video_id, platform])
        w2v_2048, bow_2048 = article_cursor.fetchone()
        data.append([
            np.frombuffer(zlib.decompress(w2v_2048), np.float32),
            np.frombuffer(zlib.decompress(bow_2048), np.float32),
            np.frombuffer(zlib.decompress(resnet_2048), np.float32),
            np.frombuffer(zlib.decompress(soundnet_1024), np.float32),
            np.frombuffer(zlib.decompress(i3d_rgb_1024), np.float32)
        ])

    pickle.dump(data, open(PICKLE_FILE, "wb+"))


if __name__ == '__main__':
    run()
