import os

# TODO remove

os.environ["PROJECT_BASE_PATH"] = os.path.dirname(os.path.realpath(__file__)) + "/../"
os.environ["MRCNN_PATH"] = os.environ["PROJECT_BASE_PATH"] + "Mask_RCNN/"
os.environ["FIGURES_PATH"] = os.environ["PROJECT_BASE_PATH"] + "reports/figures/"

os.environ["STORAGE_BASE_PATH"] = "/Volumes/DeskDrive/"
os.environ["DATA_PATH"] = os.environ["STORAGE_BASE_PATH"] + "data/"
os.environ["MODEL_PATH"] = os.environ["STORAGE_BASE_PATH"] + "models/"
