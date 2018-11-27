import zlib
import numpy as np

import psycopg2

conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
article_cursor = conn.cursor()
article_cursor.execute("SELECT count(1) FROM articles WHERE text_extraction_status='Success'")

article_cursor.execute("SELECT id, embedding FROM articles WHERE text_extraction_status='Success'")

for article_id, compressed_features in article_cursor:
    decompressed = np.frombuffer(zlib.decompress(compressed_features), np.float64)
    print(decompressed)
