from keras.layers import Dense, Dropout, Input
from keras.losses import mean_squared_error
from keras.models import Model
from keras.optimizers import RMSprop
from keras.regularizers import l2


# TODO what does the MS-postfix mean?
def build_model():
    # Building model
    input = Input(shape=(2048,))

    x = Dense(2048, activation='relu', kernel_regularizer=l2(0))(input)
    x = Dropout(0.2)(x)

    output = Dense(2048, activation='relu', kernel_regularizer=l2(0))(x)

    model = Model(inputs=[input], outputs=output)
    model.summary()

    # Compiling model
    loss = mean_squared_error
    optimizer = RMSprop(lr=0.0001)
    model.compile(loss=loss, optimizer=optimizer)

    return model

# model.load_weights(fname)
# model.to_json()
# plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)
