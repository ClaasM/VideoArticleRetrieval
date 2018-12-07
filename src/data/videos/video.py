"""
TODO don't copy-paste code from the other project
"""

import os


def get_path(platform, id):
    return os.environ["DATA_PATH"] + "/raw/videos/%s/%s.mp4" % (platform, id)

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]