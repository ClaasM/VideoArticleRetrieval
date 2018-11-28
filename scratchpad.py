import zlib

import psycopg2
import numpy as np

conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
video_cursor = conn.cursor()
video_cursor.execute("SELECT embedding FROM videos WHERE resnet_status='Success'")

for compressed_features, in video_cursor:
    decompressed = np.frombuffer(zlib.decompress(compressed_features), np.float32)
    print(decompressed)
    print(decompressed.sum())