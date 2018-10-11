import os

# TODO remove

os.environ["DATA_PATH"] = "/Volumes/DeskDrive/data/" # _temp temporarily putting it there and then manually moving it to the external drive
os.environ["MRCNN_PATH"] = "/Users/claasmeiners/PycharmProjects/video_news_classification/Mask_RCNN/"
os.environ["MODEL_PATH"] = os.path.dirname(os.path.realpath(__file__)) + "/../models/"
os.environ["FIGURES_PATH"] = os.path.dirname(os.path.realpath(__file__)) + "/../reports/figures/"
