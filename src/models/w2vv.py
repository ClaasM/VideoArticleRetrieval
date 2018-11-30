import numpy as np
from keras.losses import mean_squared_error, mean_absolute_percentage_error

np.random.seed(1337)

from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.utils import plot_model
from keras.models import Model

import keras.backend as K


# TODO what does the MS-postfix mean?
class W2VV:

    def __init__(self):
        # creat model
        print("Building model...")
        input = Input(shape=(2048,))

        x = Dense(2048, activation='relu', kernel_regularizer=l2(0))(input)
        x = Dropout(0.2)(x)

        output = Dense(2048, activation='relu', kernel_regularizer=l2(0))(x)

        self.model = Model(inputs=[input], outputs=output)
        self.model.summary()

    def compile_model(self):
        loss = mean_squared_error
        optimizer = RMSprop(lr=0.0001, clipnorm=5, rho=0.9, epsilon=1e-6)
        self.model.compile(loss=loss, optimizer=optimizer)

    def init_model(self, fname):
        self.model.load_weights(fname)

    def save_json_model(self, model_file_name):
        json_string = self.model.to_json()
        open(model_file_name, 'w+').write(json_string)

    def plot(self, filename):
        plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)

    def get_lr(self):
        # return self.model.optimizer.lr.get_value()
        return K.get_value(self.model.optimizer.lr)

    def decay_lr(self, decay=0.9):
        old_lr = self.get_lr()
        new_lr = old_lr * decay
        # new_lr = old_lr / (1 + decay*epoch)
        K.set_value(self.model.optimizer.lr, new_lr)
