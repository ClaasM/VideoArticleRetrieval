from keras.layers import Dense, Dropout, Input, regularizers
from keras.losses import mean_squared_error
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K


# TODO what does the MS-postfix mean?
def build_model(input_size, output_size):
    # Building model
    input = Input(shape=(input_size,))

    hidden = Dense(input_size,
              activation='relu',
              kernel_regularizer=regularizers.l2(0.0003))(input)
    # hidden = Dropout(0.3)(hidden)
    output = Dense(output_size,
                   activation='relu',
                   kernel_regularizer=regularizers.l2(0.0003))(hidden)

    model = Model(inputs=[input], outputs=output)
    # model.summary()

    # Compiling model
    loss = mean_squared_error
    optimizer = RMSprop()
    model.compile(loss=loss, optimizer=optimizer)

    return model


"""
Experiments with soundnet:
(new data provider)
Ls      Train   Test
0.0005  
0.0003  
0.0001  

0.0005 has better train loss AND validation loss but way worse validation median rank
"""
"""
Experiments with word2vec

Ls      Train   Test
0.001   0.235   0.235
0.0005  0.227   0.235
0.0003  0.222   0.238


With regularization 0.0005 the best median rank is 340
"""

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
