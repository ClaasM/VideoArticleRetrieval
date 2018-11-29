from keras.losses import mean_squared_error

from src.features.common import get_data
import numpy as np

import keras.backend as K

def compute_benchmark():
    data, validation_data = get_data()
    mean = np.mean([y for x, y in data])
    benchmark_loss = K.eval(K.mean(mean_squared_error([y for x, y in data], mean * len(data))))
    print("Guesing Mean MSE:")
    print(benchmark_loss) # 13051.012331730903


if __name__ == "__main__":
    # run()
    compute_benchmark()
