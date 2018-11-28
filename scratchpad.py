import zlib

import psycopg2
import numpy as np

conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
article_cursor = conn.cursor()
article_cursor.execute("SELECT embedding FROM articles LIMIT 10")

for compressed_features, in article_cursor:
    decompressed = np.frombuffer(zlib.decompress(compressed_features), np.float64)
    print(decompressed.shape)
    print(decompressed.sum())