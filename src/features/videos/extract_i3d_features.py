import time
from multiprocessing.pool import Pool

import psycopg2
import zlib

import os
import sys

import numpy as np
import cv2

from src.data.videos import video as video_helper
from src.visualization.console import CrawlingProgress

"""
Helper function to show the cropped images from numpy array.
def animate(images):
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave('./animation.gif', converted_images, fps=25)
  with open('./animation.gif','rb') as f:
      display.display(display.Image(data=f.read(), height=300))
"""
"""
def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0


# sample_video = load_video(fetch_ucf_video("v_CricketShot_g04_c02.avi"))

print("sample_video is a numpy array of shape %s." % str(sample_video.shape))

# Run the i3d model on the video and print the top 5 actions.

# First add an empty dimension to the sample video as the model takes as input
# a batch of videos.
model_input = np.expand_dims(sample_video, axis=0)

# Create the i3d model and get the action probabilities.
with tf.Graph().as_default():
  i3d = hub.Module("https://tfhub.dev/deepmind/i3d-kinetics-400/1")
  input_placeholder = tf.placeholder(shape=(None, None, 224, 224, 3), dtype=tf.float32)
  logits = i3d(input_placeholder)
  probabilities = tf.nn.softmax(logits)
  with tf.train.MonitoredSession() as session:
    [ps] = session.run(probabilities,
                       feed_dict={input_placeholder: model_input})

print("Top 5 actions:")
for i in np.argsort(ps)[::-1][:5]:
  print("%-22s %.2f%%" % (i, ps[i] * 100))
"""
'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

NUM_FRAMES = 64
NUM_SEGMENTS = 10
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2


def init_worker():
    # Need to import it here: stackoverflow.com/questions/42504669/keras-tensorflow-and-multiprocessing-in-python
    from src.features.videos.kineticsI3D import InceptionI3D
    # initialize the model
    global rgb_model
    rgb_model = InceptionI3D(
        include_top=False,
        weights='rgb_kinetics_only',
        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS))


def process(video):
    id, platform = video
    try:
        cap = cv2.VideoCapture(video_helper.get_path(platform, id))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if frame_count < NUM_FRAMES * NUM_SEGMENTS:
            return "Too few frames", id, platform, None

        # TODO this differs from approaches in current research
        # Divide the video into NUM_SEGMENTS segments and take the center NUM_FRAMES frames for analysis.
        padding_frames = int((frame_count / NUM_SEGMENTS - NUM_FRAMES) / 2)
        segments = np.zeros((NUM_SEGMENTS, NUM_FRAMES, FRAME_WIDTH, FRAME_HEIGHT, NUM_RGB_CHANNELS))
        for i in range(NUM_SEGMENTS):
            # Skip ahead padding_frames
            for _ in range(padding_frames):
                cap.read()
            # Take NUM_FRAMES frames
            for j in range(NUM_FRAMES):
                _, frame = cap.read()
                segments[i][j] = cv2.resize(video_helper.crop_center_square(frame), (FRAME_WIDTH, FRAME_HEIGHT))
            # Again, skip ahead padding_frames
            for _ in range(padding_frames):
                cap.read()
        cap.release()

        # batch size 5 allows for 4 workers on a 12GB GPU
        prediction = rgb_model.predict(np.array(segments), batch_size=5)
        # The model averages next, so I do the same. All NUM_SEGMENTS outputs are then averaged again.
        mean = prediction.mean(axis=1).mean(axis=0)[0][0]
        # Compression to reduce memory footprint of sparse vectors
        return "Success", id, platform, zlib.compress(mean, 9)
    except Exception as e:
        return str(e), id, platform, None


# TODO in thesis: this is from Learning Joint Embedding with Multimodal Cues for Cross-Modal Video-Text Retrieval
# TODO maybe use from deepmind repository: https://github.com/deepmind/kinetics-i3d/tree/master/data
# TODO set image_data_format='channels_last
def run():
    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    video_cursor = conn.cursor()
    update_cursor = conn.cursor()
    video_cursor.execute("SELECT id, platform FROM videos WHERE i3d_rgb_status<>'Success' AND platform='facebook'")
    videos = video_cursor.fetchall()
    crawling_progress = CrawlingProgress(len(videos), update_every=100)
    # 4 works best. Too many and each worker doesn't have the GPU memory it needs
    with Pool(4, initializer=init_worker) as pool:
        for status, id, platform, compressed_feature in pool.imap_unordered(process, videos, chunksize=10):
            if status == 'Success':
                # Insert embedding and update the classification status
                update_cursor.execute(
                    "UPDATE videos SET i3d_rgb_status = 'Success', i3d_rgb_1024 = %s WHERE id=%s AND platform=%s",
                    [compressed_feature, id, platform])
            else:
                update_cursor.execute(
                    "UPDATE videos SET i3d_rgb_status = %s WHERE id=%s AND platform=%s",
                    [status, id, platform])
            conn.commit()
            crawling_progress.inc()


if __name__ == '__main__':
    run()
