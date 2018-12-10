"""
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
"""

from keras.layers import Dense, Input, regularizers
from keras.losses import mean_squared_error
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K
import tensorflow as tf
import numpy as np


def build_model(input_size, output_size):
    # Building model
    print("Input size: %d, Output size: %d" % (input_size, output_size))
    input = Input(shape=(input_size,))

    hidden = Dense(max(output_size, input_size),
                   activation='relu',
                   kernel_regularizer=regularizers.l2(0.00005))(input)
    # hidden = Dropout(0.3)(hidden)
    output = Dense(output_size,
                   activation='relu',
                   kernel_regularizer=regularizers.l2(0.00005))(hidden)

    model = Model(inputs=[input], outputs=output)
    # model.summary()

    # Compiling model
    # TODO can't really use a distance metric here because
    # we need a batch x 2048, batch x 2048 -> 2048 function
    # Cosine distance is batch x 2048, batch x 2048 -> batch x batch
    loss = cosine_proximity
    optimizer = RMSprop()
    model.compile(loss=loss, optimizer=optimizer)

    return model


# TODO move to different files since its used in the ranking callback too
def cosine_proximity(y_true, y_pred):
    y_true = tf.nn.l2_normalize(y_true, dim=-1)
    y_pred = tf.nn.l2_normalize(y_pred, dim=-1)
    return -K.sum(y_true * y_pred, axis=-1)


if __name__ == '__main__':
    test_1 = np.array([[1., 2.]]) # 20
    test_2 = np.array([[1., 2.], [3., 4.], [5, 1]])
    print(K.eval(cosine_proximity(
        test_1,
        test_2,
    )))

    """
    Experiments with soundnet:
    (new data provider)
    Ls      Train   Test    r10
    0.0005  
    0.0003  0.145   0.151   0.243
    0.0001  0.194   0.168   0.278
    
    0.0005 has better train loss AND validation loss but way worse validation median rank
    """ \
    """
    Experiments with word2vec
    
    Ls      Train   Test
    0.001   0.235   0.235
    0.0005  0.227   0.235
    0.0003  0.222   0.238
    
    
    With regularization 0.0005 the best median rank is 340
    """ \
    """
    Optimizing Test loss.
    Procedure: Finding a regularization with good train score and not too much overfitting.
    Then fix overfitting with Dropout.
    
    Best test loss
    
    L2 beta, in 1/10^n steps, no dropout:
    Ls      Train   Test
    0.1:    0.455   0.451
    0.01:   0.263   0.260
    0.001:  0.239   0.238
    0.0005: 0.233   0.234
    0.0003: 0.226   0.234
    0.0001: 0.215   0.270
    0.00005:0.187   0.283
    0.00003:0.171   0.303
    0.00001:0.134   0.352
    
    --> Going with 0.0001
    
    Dropout Train   Test
    0.1:    0.211   0.266
    0.2:    0.214   0.246
    0.3:    0.216   0.246
    0.4:    0.219   0.244
    
    --> Going with 0.00005, since we have to beat 0.236, and 0.0001 isn't cutting it
    0.1:    0.193   0.273
    0.2:    0.194   0.267
    0.3:    0.201   0.263
    
    --> None of this pushes the test score above what we got with regularization 0.001
    
    With regularization 0.0005/0.0003 the best median rank is 400 both times.
    
    TODO put this into the thesis?
    """

    # model.load_weights(fname)
    # model.to_json()
    # plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)
