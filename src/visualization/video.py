from src.data.videos import video as video_helper

from IPython.display import HTML, display

import os
import base64


def display_video(platorm="facebook", id="CadburyBournvita/1937970696267088"):
    path = video_helper.get_path(platform=platorm, id=id)
    # path = os.path.relpath(path, os.getcwd())
    video_encoded = base64.b64encode(open(path, "rb").read())
    display(HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(video_encoded.decode('ascii'))))
