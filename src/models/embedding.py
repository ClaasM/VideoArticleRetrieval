from keras.layers import Dense, Dropout, Input, regularizers
from keras.losses import mean_squared_error
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K


# TODO what does the MS-postfix mean?
def build_model():
    # Building model
    input = Input(shape=(2048,))

    x = Dense(2048,
              activation='relu',
              kernel_regularizer=regularizers.l2(0.0001))(input)  # , kernel_regularizer=l2(0)
    x = Dropout(0.4)(x)
    output = Dense(2048,
                   activation='relu',
                   kernel_regularizer=regularizers.l2(0.0001))(x)  # , kernel_regularizer=l2(0)

    model = Model(inputs=[input], outputs=output)
    model.summary()

    # Compiling model
    loss = mean_squared_error
    optimizer = RMSprop(lr=0.0001)
    model.compile(loss=loss, optimizer=optimizer)

    return model
"""
Observations:
Higher Dropout, lower overfitting, worse loss, worse r1, rest the same
"""



# model.load_weights(fname)
# model.to_json()
# plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)
