import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from src.models.resnet_152 import ResNet152

# Classify the image at
image = "/Users/claasmeiners/data/examples/images/duck.jpg"

#prototxt = "/Users/claasmeiners/models/ResNet-152-deploy.prototxt"
#caffemodel = "/Users/claasmeiners/models/ResNet-152-model.caffemodel"
#binaryproto = "/Users/claasmeiners/models/ResNet_mean.binaryproto"

# create model
model = ResNet152()


# define function for input preprocessing
def preprocess(x):
    x = resize(x, (224, 224), mode='constant') * 255
    x = preprocess_input(x)
    if x.ndim == 3:
        x = np.expand_dims(x, 0)
    return x


# prepare image
img = imread('./imgs/cat.jpg')
x = preprocess(img)

# make prediction and decode it
y = model.predict(x)
pred_title = decode_predictions(y, top=1)[0][0][1]

# print result
print(pred_title)
### tiget_cat
