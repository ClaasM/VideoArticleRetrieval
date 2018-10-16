import os
from pocketsphinx import AudioFile
from pocketsphinx import Pocketsphinx
from src import util

test_video = os.environ['DATA_PATH'] + "/other/sphinx_test_video/beachball.mp4"
test_audio = os.environ['DATA_PATH'] + "/other/sphinx_test_audio/interview.wav"


fps = 100  # default
audio_file = AudioFile(audio_file=test_audio, frate=100)
for phrase in audio_file:  # frate (default=100)

    print(" ".join([s.word for s in phrase.seg()]))


#print(Pocketsphinx().decode(audio_file=test_audio))


# i'm home i'm a i'm won't home all cool i'm wall long and lulu move up at bay back when when i'm i'm home
