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
from src.visualization.console import StatusVisualization

from keras.applications.imagenet_utils import preprocess_input

# Force CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

EVERY_FRAME = 30
MIN_IMAGES = 10
# Total: Minimum 300 Frames

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
                x = resize(image, (224, 224), mode='constant') * 255
                x = preprocess_input(x)
                images.append(x)
            count += 1
        else:
            # Reached the end of the video
            cap.release()
            break

    if len(images) > MIN_IMAGES:
        # Batch predict
        frame_results = model.predict(np.array(images))
        # The shape is (n_frames, 1, 1, layer_output)
        frame_results = frame_results.reshape(-1, frame_results.shape[-1])
        # Mean pooling
        mean = np.mean(frame_results, axis=0)
        # Compression to reduce memory footprint of sparse vectors
        return "Success", id, platform, zlib.compress(mean, 9)
    else:
        return "Too few frames", id, platform, None


def run():
    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    video_cursor = conn.cursor()
    update_cursor = conn.cursor()
    video_cursor.execute("SELECT id, platform FROM videos WHERE resnet_status<>'Success'")
    videos = video_cursor.fetchall()
    crawling_progress = StatusVisualization(len(videos), update_every=100)
    # 4 works best. Too many and each worker doesn't have the GPU memory it needs
    with Pool(4, initializer=init_worker) as pool:
        for status, id, platform, compressed_features in pool.imap_unordered(process, videos, chunksize=10):
            print()
            if status == 'Success':
                # Insert embedding and update the classification status
                update_cursor.execute(
                    "UPDATE videos SET resnet_status = 'Success', resnet_2048 = %s WHERE id=%s AND platform=%s",
                    [compressed_features, id, platform])
            else:
                update_cursor.execute(
                    "UPDATE videos SET resnet_status = %s WHERE id=%s AND platform=%s",
                    [status, id, platform])
            conn.commit()
            crawling_progress.inc()


if __name__ == '__main__':
    run()
