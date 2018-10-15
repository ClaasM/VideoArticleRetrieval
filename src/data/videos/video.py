"""
Inspired by:
https://github.com/tbhaxor/fbdown
TODO don't copy-paste code from the other project
"""

import os
import re
import traceback
import urllib.parse

import requests
from requests import HTTPError

from src.data.videos import video as video_helper


def get_path(platform, id):
    return os.environ["DATA_PATH"] + "/raw/videos/%s/%s.mp4" % (platform, id)



if __name__ == '__main__':
    video_id = get_id_from_url(embedding_url)
    video = download(video_id)
    print(video)
