import subprocess
import uuid
import zlib
from multiprocessing.pool import Pool

import librosa
import numpy as np
import os
import psycopg2
import tempfile

from src.data.videos import video as video_helper
from src.visualization.console import CrawlingProgress


def extract_audio(video_path):
    output_path = tempfile.gettempdir() + ("/%s.wav" % uuid.uuid4())
    command = "ffmpeg -i %s -acodec pcm_f32le -ac 1 -ar 22050 -y -loglevel quiet %s" % (video_path, output_path)
    subprocess.call(command.split(" "))
    return output_path


def init_worker():
    # Need to import it here: stackoverflow.com/questions/42504669/keras-tensorflow-and-multiprocessing-in-python
    from src.features.videos.soundnet import build_model
    # initialize the model
    global model
    model = build_model()


def process(video):
    # Takes a video and returns every nth frame preprocessed as a numpy-array
    id, platform = video
    try:
        path = extract_audio(video_helper.get_path(platform, id))
        audio, _ = librosa.load(path, dtype='float32', sr=22050, mono=True)

        # SoundNet needs the range to be between -256 and 256
        # In addition to the research this is based on, we scale the amplitude
        maximum = max(audio.max(), -audio.min())
        if maximum != 0.0:
            audio *= 256.0 / maximum
            # reshaping the audio data so it fits into the graph (batch_size, num_samples, num_filter_channels)
            audio = np.reshape(audio, (1, -1, 1))
            prediction = model.predict(audio)
            subprocess.call(["rm", path])
            prediction = prediction.mean(axis=1)[0]
            return "Success", id, platform, zlib.compress(prediction, 9)
        else:
            subprocess.call(["rm", path])
            return "No Audio", id, platform, None
    except Exception as e:
        return str(e), id, platform, None


def run():
    conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
    video_cursor = conn.cursor()
    update_cursor = conn.cursor()
    # TODO do for all
    video_cursor.execute("SELECT id, platform FROM videos WHERE "
                         "soundnet_status<>'Success' AND soundnet_status<>'No Audio' AND resnet_status='Success'")
    videos = video_cursor.fetchall()
    crawling_progress = CrawlingProgress(len(videos), update_every=100)
    # 4 works best. Too many and each worker doesn't have the GPU memory it needs
    with Pool(3, initializer=init_worker) as pool:
        for status, id, platform, compressed_features in pool.imap_unordered(process, videos, chunksize=10):
            if status == "Success":
                # Insert embedding and update the classification status
                update_cursor.execute(
                    "UPDATE videos SET soundnet_status = 'Success', soundnet_1024=%s WHERE id=%s AND platform=%s",
                    [compressed_features, id, platform])
            else:
                print(str(status)[:100])
                update_cursor.execute(
                    "UPDATE videos SET soundnet_status=%s WHERE id=%s AND platform=%s",
                    [status, id, platform])
            conn.commit()
            crawling_progress.inc()


if __name__ == '__main__':
    run()
    # TODO run with more memory for those long audio tracks
