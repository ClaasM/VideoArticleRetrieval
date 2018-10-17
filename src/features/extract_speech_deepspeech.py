import os
from pocketsphinx import AudioFile
from pocketsphinx import Pocketsphinx
from src import util


test_audio = os.environ['DATA_PATH'] + "/other/sphinx_test_audio/interview_excerpt.raw"


test_audio = "/Volumes/DeskDrive/data/other/sphinx_test_audio/interview_mono_nonoise.wav"
# deepspeech --model models/deepspeech/output_graph.pbmm --alphabet models/deepspeech/alphabet.txt --lm models/deepspeech/lm.binary --trie models/deepspeech/trie --audio /Volumes/DeskDrive/data/other/sphinx_test_audio/interview_mono_nonoise.wav
