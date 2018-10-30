"""
Set a bunch of env-vars if they're not set yet
"""

import os

if "STORAGE_BASE_PATH" not in os.environ:
    os.environ["STORAGE_BASE_PATH"] = "/Volumes/DeskDrive/"

os.environ["DATA_PATH"] = os.environ["STORAGE_BASE_PATH"] + "data/"
os.environ["MODEL_PATH"] = os.environ["STORAGE_BASE_PATH"] + "models/"

os.environ["PROJECT_BASE_PATH"] = os.path.dirname(os.path.realpath(__file__)) + "/../"
os.environ["FIGURES_PATH"] = os.environ["PROJECT_BASE_PATH"] + "/reports/figures/"

if "MRCNN_PATH" not in os.environ:
    os.environ["MRCNN_PATH"] = os.environ["PROJECT_BASE_PATH"] + "/Mask_RCNN/"
if "DARKNET_PATH" not in os.environ:
    os.environ["DARKNET_PATH"] = os.environ["PROJECT_BASE_PATH"] + '/darknet/'


