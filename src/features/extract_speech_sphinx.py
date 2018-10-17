import os
from pocketsphinx import AudioFile
from pocketsphinx import Pocketsphinx
from src import util

#test_video = os.environ['DATA_PATH'] + "/other/sphinx_test_video/beachball.mp4"
test_audio = os.environ['DATA_PATH'] + "/other/sphinx_test_audio/interview_excerpt.raw"


fps = 100  # default
audio_file = AudioFile(audio_file=test_audio, frate=100)
for phrase in audio_file:  # frate (default=100)

    print(" ".join([s.word for s in phrase.seg()]))


#print(Pocketsphinx().decode(audio_file=test_audio))


# Supported formats:
# RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz (or 8000 hz)


# Change sampling rate
# ffmpeg -i interview.wav -ar 16000 interview_16kHz.wav
# convert to mono:


# sox interview_excerpt_16kHz.wav -b 16 -s -c 1 -r 16k -t raw interview_excerpt.raw