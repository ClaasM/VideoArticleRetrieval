import zlib

import psycopg2
import numpy as np

conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
video_cursor = conn.cursor()
update_cursor = conn.cursor()
video_cursor.execute("SELECT soundnet_1024 FROM videos WHERE soundnet_status='Success'")
for feature, in video_cursor:
    print(np.frombuffer(zlib.decompress(feature), np.float32).sum())