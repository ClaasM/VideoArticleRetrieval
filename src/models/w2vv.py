import numpy as np

np.random.seed(1337)

from keras.layers import Dense, Dropout, Input, concatenate
from keras.optimizers import SGD, Adagrad, RMSprop, Adam
from keras.regularizers import l2
from keras.utils import plot_model
from keras.models import Model

import keras.backend as K


# basic word2visualvec
class Base_model:
    def compile_model(self):

        clipnorm = 5
        optimizer = 'rmsprop'
        loss = 'mse'
        learning_rate = 0.0001

        if optimizer == 'sgd':
            # let's train the model using SGD + momentum (how original).
            if clipnorm > 0:
                sgd = SGD(lr=learning_rate, clipnorm=clipnorm, decay=1e-6, momentum=0.9, nesterov=True)
            else:
                sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(loss=loss, optimizer=sgd)
        elif optimizer == 'rmsprop':
            if clipnorm > 0:
                rmsprop = RMSprop(lr=learning_rate, clipnorm=clipnorm, rho=0.9, epsilon=1e-6)
            else:
                rmsprop = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
            self.model.compile(loss=loss, optimizer=rmsprop)
        elif optimizer == 'adagrad':
            if clipnorm > 0:
                adagrad = Adagrad(lr=learning_rate, clipnorm=clipnorm, epsilon=1e-06)
            else:
                adagrad = Adagrad(lr=learning_rate, epsilon=1e-06)
            self.model.compile(loss=loss, optimizer=adagrad)
        elif optimizer == 'adma':
            if clipnorm > 0:
                adma = Adam(lr=learning_rate, clipnorm=clipnorm, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            else:
                adma = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            self.model.compile(loss=loss, optimizer=adma)

    def init_model(self, fname):
        self.model.load_weights(fname)

    def save_json_model(self, model_file_name):
        json_string = self.model.to_json()
        if model_file_name[-5:] != '.json':
            model_file_name = model_file_name + '.json'
        open(model_file_name, 'w').write(json_string)

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


# TODO what does the MS-postfix mean?
class W2VV_MS(Base_model):
    def __init__(self, n_layers):
        # creat model
        print("Building model...")

        # bow, word2vec or word hashing embedded sentence vector
        input_layer = Input(shape=(n_layers[0],))

        x = input_layer
        for n_neuron in range(1, len(n_layers) - 1):
            x = Dense(n_layers[n_neuron], activation='relu', kernel_regularizer=l2(0))(x)
            x = Dropout(0.2)(x)

        output = Dense(n_layers[-1], activation='relu', kernel_regularizer=l2(0))(x)

        self.model = Model(inputs=[input_layer], outputs=output)
        self.model.summary()

    def predict_one(self, text_vec, text_vec_2):
        text_embed_vec = self.model.predict([np.array([text_vec]), np.array([text_vec_2])])
        return text_embed_vec[0]

    def predict_batch(self, text_vec_batch):
        # TODO maybe this doesn't need to be wrapped in an array again
        text_embed_vecs = self.model.predict([np.array(text_vec_batch)])
        # TODO make this obsolete
        return text_embed_vecs
