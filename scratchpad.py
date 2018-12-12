import keras.backend as K
import numpy as np
from keras import Model, Input
from keras.callbacks import Callback
from keras.layers import Dense, regularizers
from keras.losses import mean_squared_error
from keras.optimizers import RMSprop

x_train = np.random.random((10000, 1024)).astype(np.float32)
y_train = np.random.random((10000, 1024)).astype(np.float32)
x_validation = np.random.random((1000, 1024)).astype(np.float32)
y_validation = np.random.random((1000, 1024)).astype(np.float32)


class ValidationCallback(Callback):
    def __init__(self, validation_x, validation_y):
        super(ValidationCallback, self).__init__()
        self.validation_x = validation_x
        self.validation_y = validation_y

    def on_epoch_end(self, epoch, logs=None):
        # What am I missing in this loss calculation that keras is doing?
        validation_y_predicted = self.model.predict(self.validation_x)

        # Compute regularization term for each layer
        weights = self.model.trainable_weights
        reg_term = 0
        for i, w in enumerate(weights):
            if i % 2 == 0:  # weights from layer i // 2
                w_f = K.flatten(w)
                reg_term += 0.01 * K.sum(K.square(w_f))

        mse_loss = K.mean(mean_squared_error(self.validation_y, validation_y_predicted))
        loss = float(K.eval(mse_loss + reg_term))

        print("%.4f" % K.eval(K.mean(mean_squared_error(self.validation_y, validation_y_predicted))))
        print("%.4f" % loss)


input = Input(shape=(1024,))
hidden = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(input)
output = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(hidden)

model = Model(inputs=[input], outputs=output)

optimizer = RMSprop()
model.compile(loss='mse', optimizer=optimizer)

model.fit(x=x_train,
          y=y_train,
          callbacks=[ValidationCallback(x_validation, y_validation)],
          validation_data=(x_validation, y_validation))


print(model.evaluate(x_validation, y_validation))
