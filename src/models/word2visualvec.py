import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from src.features.videos.resnet_152 import ResNet152

from keras.preprocessing import image

# Classify the image at
img_path = "/Users/claasmeiners/data/examples/images/cat.jpg"


# define function for input preprocessing
"""
def preprocess(x):
    x = resize(x, (224, 224), mode='constant') * 255
    x = preprocess_input(x)
    if x.ndim == 3:
        x = np.expand_dims(x, 0)
    return x
"""

model = ResNet152(include_top=False, weights='imagenet')

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
