"""
TODO don't copy-paste code from the other project
"""

import os


def get_path(platform, id):
    return os.environ["DATA_PATH"] + "/raw/videos/%s/%s.mp4" % (platform, id)
