import zlib

import psycopg2
import numpy as np
import cv2
from src.data.videos import video as video_helper

conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
video_cursor = conn.cursor()  # for the videos
video_cursor.execute("SELECT id, platform, resnet_2048, soundnet_1024, i3d_rgb_1024 FROM videos "
                     "WHERE resnet_status='Success' "
                     "AND soundnet_status='Success' "
                     "AND i3d_rgb_status='Success' "
                     "ORDER BY random()")
for (video_id, platform, resnet_compressed, soundnet_compressed, i3d_rgb_compressed) in video_cursor:
    test = np.frombuffer(zlib.decompress(i3d_rgb_compressed), np.float32)
    print(test.sum())
    print(test.mean())