"""
Several Feature extraction methods for videos involving resnet-152.
Saves the features in the format required or w2vv.
  AND (ctid::text::point)[1]::INT % 4 = 3
"""
import zlib
from multiprocessing.pool import Pool

import cv2
import numpy as np
import psycopg2

from skimage.transform import resize

from src.data.videos import video as video_helper
from src.visualization.console import CrawlingProgress

from keras.applications.imagenet_utils import preprocess_input

# Force CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""
This might be useful for other models:
layer_name = 'avg_pool'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
"""

EVERY_FRAME = 30


def init_worker():
    # Need to import it here: stackoverflow.com/questions/42504669/keras-tensorflow-and-multiprocessing-in-python
    # from keras.applications import ResNet50
    from src.features.videos.resnet_152 import ResNet152
    # initialize the model
    global model
    model = ResNet152(include_top=False)


def process(video):
    # Takes a video and returns every nth frame preprocessed as a numpy-array
    id, platform = video

    images = []
    cap = cv2.VideoCapture(video_helper.get_path(platform, id))
    count = 0
    while True:
        success, image = cap.read()
        if success:
            if count % EVERY_FRAME == 0:
                # TODO choose random x/y-location if aspect ratio is not square
                x = resize(image, (224, 224), mode='constant') * 255
                x = preprocess_input(x)
                images.append(x)
            count += 1
        else:
            # Reached the end of the video
            break

    # Batch predict
    frame_results = model.predict(np.array(images))
    # The shape is (n_frames, 1, 1, layer_output)
    frame_results = frame_results.reshape(-1, frame_results.shape[-1])
    # Mean pooling
    mean = np.mean(frame_results, axis=0)
    # Compression to reduce memory footprint of sparse vectors
    return id, platform, zlib.compress(mean, 9)


def run():
    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    video_cursor = conn.cursor()
    update_cursor = conn.cursor()
    video_cursor.execute("SELECT id, platform FROM videos WHERE resnet_status<>'Success'")
    videos = video_cursor.fetchall()
    crawling_progress = CrawlingProgress(len(videos), update_every=100)
    # 4 works best. Too many and each worker doesn't have the GPU memory it needs
    with Pool(4, initializer=init_worker) as pool:
        for id, platform, compressed_features in pool.imap_unordered(process, videos, chunksize=10):
            # Insert embedding and update the classification status
            update_cursor.execute(
                "UPDATE videos SET resnet_status = 'Success', embedding=%s WHERE id=%s AND platform=%s",
                [compressed_features, id, platform])
            conn.commit()
            crawling_progress.inc()


if __name__ == '__main__':
    run()
