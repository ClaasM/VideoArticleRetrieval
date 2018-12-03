import zlib

import psycopg2
import numpy as np
import cv2
from src.data.videos import video as video_helper

conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
video_cursor = conn.cursor()
update_cursor = conn.cursor()
video_cursor.execute("SELECT platform, id FROM videos WHERE resnet_status='Success'")
for index, (platform, id) in enumerate(video_cursor):
    cap = cv2.VideoCapture(video_helper.get_path(platform, id))
    if index % 1000 == 0:
        print(index)
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) < 300:
        print("Too few!")
        update_cursor.execute(
            "UPDATE videos SET resnet_status=%s WHERE id=%s AND platform=%s",
            ["Too few frames", id, platform])
        conn.commit()