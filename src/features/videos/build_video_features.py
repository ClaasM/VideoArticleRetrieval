"""
Several Feature extraction methods for videos involving resnet-152.
Saves the features in the format required or w2vv.
"""

import zlib

from src.features.videos.resnet_152 import ResNet152

# Force CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tempfile

import cv2
import numpy as np
import psycopg2
from keras.applications.imagenet_utils import preprocess_input
from skimage.io import imread
from skimage.transform import resize

from src.data.videos import video as video_helper
from src.visualization.console import CrawlingProgress

"""
This might be useful for other models:
layer_name = 'avg_pool'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
"""

EVERY_FRAME = 30

if __name__ == '__main__':

    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    video_cursor = conn.cursor()
    update_cursor = conn.cursor()
    video_cursor.execute("SELECT id, platform FROM videos WHERE resnet_status<>'Success' AND platform = 'facebook'")
    videos = video_cursor.fetchall()
    model = ResNet152(include_top=False)
    crawling_progress = CrawlingProgress(len(videos), update_every=10)
    for id, platform in videos:

        # We need to extract the images first
        images = []
        cap = cv2.VideoCapture(video_helper.get_path(platform, id))
        count = 0
        while True:
            success, image = cap.read()
            if success:
                if count % EVERY_FRAME == 0:
                    path = tempfile.gettempdir() + "/%09d.jpg" % count
                    cv2.imwrite(path, image)
                    images.append(path)
                count += 1
            else:
                # Reached the end of the video
                break

        image_results = list()
        for index, image_path in enumerate(images):
            x = imread(image_path)
            # TODO choose random x/y-location if aspect ratio is not square
            x = resize(x, (224, 224), mode='constant') * 255
            x = preprocess_input(x)
            x = np.expand_dims(x, 0)  # Flatten R,G,B
            y = model.predict(x)
            image_results.append(y[0][0][0])

        # Mean pooling
        mean = np.mean(image_results, axis=0)
        compressed_features = zlib.compress(np.array(mean), 9)

        # Update the classification status
        update_cursor.execute("UPDATE videos SET resnet_status = 'Success', embedding=%s WHERE id=%s AND platform=%s",
                              [compressed_features, id, platform])
        conn.commit()
        crawling_progress.inc()


