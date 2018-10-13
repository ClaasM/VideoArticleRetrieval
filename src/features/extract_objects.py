"""
Make sure Mask R-CNN is in your python path.
TODO put this into the documentation
"""

# TODO every one second, sum up all detected objects (maybe even multiplied by size) and divide by the number of seconds

from src import constants

import os, sys

sys.path += [os.environ["MRCNN_PATH"]]

import mrcnn.model as modellib
from samples.coco import coco
import skimage
import numpy as np
import os
import cv2




class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # How many images can fit into memory?


config = InferenceConfig()
# config.display()
MODEL_DIR = os.environ["MRCNN_PATH"] + "/logs"
COCO_MODEL_PATH = os.environ["MRCNN_PATH"] + "/mask_rcnn_coco.h5"

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

videos = [os.environ["DATA_PATH"] + "examples/videos/who_is_america.mp4"]

counter = 0
for vid in videos:
    images = []
    cap = cv2.VideoCapture(vid)
    count = 0
    success = True
    while success:
        success, image = cap.read()
        if count % 30 == 0:
            images.append(image)
        count += 1

    print(len(images))
    #r = model.detect([images, verbose=1)[0]
    #images = [os.environ["DATA_PATH"] + "examples/images/dog_bike.jpg"]

    for index, image in enumerate(images):
        # TODO it might be faster to predict all frames of a video at once
        r = model.detect([image], verbose=1)[0]

        n_instances = r['rois'].shape[0]
        height, width = image.shape[:2]

        print("%d: Found %d rois in an image sized %dx%d" % (index, n_instances, width, height))

        for i in range(n_instances):
            if not np.any(r['rois'][i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = r['rois'][i]
            class_id = r['class_ids'][i]
            score = r['scores'][i]
            label = constants.coco_class_names[class_id]
            print("%d,%d to %d,%d: %s (%.3f)" % (x1, y1, x2, y2, label, score))

# TODO make MaskRCNN code redundant

