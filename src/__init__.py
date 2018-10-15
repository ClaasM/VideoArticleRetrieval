import os

# TODO remove

os.environ["BASE_PATH"] = os.path.dirname(os.path.realpath(__file__)) + "/../"
os.environ["MRCNN_PATH"] = os.environ["BASE_PATH"] + "Mask_RCNN/"
os.environ["MODEL_PATH"] = os.environ["BASE_PATH"] + "models/"
os.environ["FIGURES_PATH"] = os.environ["BASE_PATH"] + "reports/figures/"

os.environ["DATA_PATH"] = "/Volumes/DeskDrive/data/"
