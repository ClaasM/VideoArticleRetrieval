import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
import keras.backend as K
K.set_session(sess)

import numpy as np
from keras import Model
from keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, InputLayer
from keras.models import Sequential

WEIGHTS_PATH = os.environ["MODEL_PATH"] + "/sound8.npy"

def build_model():
    """
    Builds up the SoundNet model and loads the weights from a given model file (8-layer model is kept at models/sound8.npy).
    :return:
    """
    model_weights = np.load(WEIGHTS_PATH, encoding='latin1').item()
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, None, 1)))

    filter_parameters = [
        {'name': 'conv1', 'num_filters': 16, 'padding': 32,
         'kernel_size': 64, 'conv_strides': 2,
         'pool_size': 8, 'pool_strides': 8},

        {'name': 'conv2', 'num_filters': 32, 'padding': 16,
         'kernel_size': 32, 'conv_strides': 2,
         'pool_size': 8, 'pool_strides': 8},

        {'name': 'conv3', 'num_filters': 64, 'padding': 8,
         'kernel_size': 16, 'conv_strides': 2},

        {'name': 'conv4', 'num_filters': 128, 'padding': 4,
         'kernel_size': 8, 'conv_strides': 2},

        {'name': 'conv5', 'num_filters': 256, 'padding': 2,
         'kernel_size': 4, 'conv_strides': 2,
         'pool_size': 4, 'pool_strides': 4},

        {'name': 'conv6', 'num_filters': 512, 'padding': 2,
         'kernel_size': 4, 'conv_strides': 2},

        {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
         'kernel_size': 4, 'conv_strides': 2},

        {'name': 'conv8_2', 'num_filters': 401, 'padding': 0,
         'kernel_size': 8, 'conv_strides': 2},
    ]

    for x in filter_parameters:
        model.add(ZeroPadding1D(padding=x['padding']))
        model.add(Conv1D(x['num_filters'],
                         kernel_size=x['kernel_size'],
                         strides=x['conv_strides'],
                         padding='valid'))
        weights = model_weights[x['name']]['weights'].reshape(model.layers[-1].get_weights()[0].shape)
        biases = model_weights[x['name']]['biases']

        model.layers[-1].set_weights([weights, biases])

        if 'conv8' not in x['name']:
            gamma = model_weights[x['name']]['gamma']
            beta = model_weights[x['name']]['beta']
            mean = model_weights[x['name']]['mean']
            var = model_weights[x['name']]['var']

            model.add(BatchNormalization())
            model.layers[-1].set_weights([gamma, beta, mean, var])
            model.add(Activation('relu'))
        if 'pool_size' in x:
            model.add(MaxPooling1D(pool_size=x['pool_size'],
                                   strides=x['pool_strides'],
                                   padding='valid'))

    #
    return Model(inputs=model.input, outputs=model.get_layer('activation_7').output)