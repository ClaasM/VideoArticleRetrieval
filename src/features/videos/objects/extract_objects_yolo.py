"""
Make sure DYLD_LIBRARY_PATH is set
"""

import os
import tempfile

import psycopg2
import cv2
import time

from src.models import darknet_wrapper
from src.data.videos import video as video_helper

from src.visualization.console import StatusVisualization


def run():
    MODEL = "yolov3"  # Postfix -tiny
    net, meta = darknet_wrapper.initialize_classifier(
        config="cfg/%s.cfg" % MODEL,
        weights="weights/%s.weights" % MODEL,
        data="cfg/coco.data")

    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    c = conn.cursor()
    # Just classifying facebook videos for now
    c.execute("SELECT id, platform FROM videos WHERE object_detection_yolo_status<>'Success' AND platform = 'facebook'")
    videos = c.fetchall()

    print("%d videos left to analyze" % len(videos))

    crawling_progress = StatusVisualization(len(videos), update_every=10)
    for id, platform in videos:
        # print(platform, id)
        # We need to extract the images first
        # start = time.time()
        images = []
        cap = cv2.VideoCapture(video_helper.get_path(platform, id))
        count = 0
        while True:
            success, image = cap.read()
            if success:
                if count % 30 == 0:
                    path = tempfile.gettempdir() + "/%05d.jpg" % count
                    cv2.imwrite(path, image)
                    images.append(path)
                count += 1
            else:
                # Reached the end of the video
                break

        # print("Extracted %d images in %d seconds" % (len(images), time.time() - start))
        # start = time.time()

        for index, image in enumerate(images):
            try:
                result = darknet_wrapper.detect(net, meta, image)

                # print("%d: Found %d rois in %s" % (index, len(result), image))
                for entity in result:
                    # format is (class, probability (x,y,width, height)) ANKERED IN THE CENTER!
                    (label, probability, (x, y, width, height)) = entity
                    # x,y,height and width are not saved for now.
                    # print("%d,%d (%dx%d): %s (%.3f)" % (x, y, width, height, label, probability))
                    c.execute(
                        "INSERT INTO object_detection_yolo(id,platform,second,class,probability) VALUES (%s,%s,%s,%s,%s)",
                        [id, platform, index, str(label, "utf-8"), probability])
                    conn.commit()
            except Exception as e:
                print(e)

        # Update the classification status
        c.execute(
            "UPDATE videos SET object_detection_yolo_status = 'Success' WHERE id=%s AND platform=%s",
            [id, platform])
        conn.commit()
        # print("Detection took %d seconds" % (time.time() - start))
        crawling_progress.inc()


if __name__ == "__main__":
    run()

"""
TODO
Some statistics:
Average number of items per video:
"""
