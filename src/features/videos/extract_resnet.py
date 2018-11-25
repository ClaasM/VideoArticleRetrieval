"""
Several Feature extraction methods for videos involving resnet-152.
Saves the features in the format required or w2vv.
"""
"""
# Force CPU for now
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
"""

import tempfile

import cv2
import numpy as np
import psycopg2
from keras.applications.imagenet_utils import preprocess_input
from skimage.io import imread
from skimage.transform import resize

from src.data.videos import video as video_helper
from src.features.videos.resnet_152 import ResNet152
from src.visualization.console import CrawlingProgress


def preprocess(x):
    x = resize(x, (224, 224), mode='constant') * 255
    x = preprocess_input(x)
    if x.ndim == 3:
        x = np.expand_dims(x, 0)
    return x


if __name__ == '__main__':

    model = ResNet152(include_top=False)

    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    c = conn.cursor()
    c.execute("SELECT id, platform FROM videos WHERE resnet_status<>'Success' AND platform = 'facebook'")
    videos = c.fetchall()
    crawling_progress = CrawlingProgress(len(videos), update_every=10)
    for id, platform in videos:

        # We need to extract the images first
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

        print(len(images))

        for index, image_path in enumerate(images):
            # prepare image
            img = imread(image_path)
            x = preprocess(img)
            y = model.predict(x)

            # print result
            print("Result: %s" % y)
            ### tiget_cat

        # Update the classification status
        # c.execute("UPDATE videos SET resnet_status = 'Success' WHERE id=%s AND platform=%s", [id, platform])
        # conn.commit()
        crawling_progress.inc()
