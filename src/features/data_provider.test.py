import time
import numpy as np
from src.features.data_provider import DataProvider

start = time.time()

data_provider = DataProvider(1000, 1000)

print(time.time() - start)

print(data_provider.validation_y.shape)
print(data_provider.validation_x.shape)
print(data_provider.test_y.shape)
print(data_provider.test_x.shape)
print(data_provider.train_x.shape)
print(data_provider.train_y.shape)

# Should be
# (1000, 2048)
# (1000, 2048)
# (1000, 2048)
# (1000, 2048)
# (52218, 2048)
# (52218, 2048)

# TODO also validate mean sums
print(np.sum(np.mean(data_provider.validation_y, axis=0)))
print(np.sum(np.mean(data_provider.test_y, axis=0)))
print(np.sum(np.mean(data_provider.train_y, axis=0)))
print(np.sum(np.mean(data_provider.validation_x, axis=0)))
print(np.sum(np.mean(data_provider.test_x, axis=0)))
print(np.sum(np.mean(data_provider.train_x, axis=0)))

# 161.81772583182868
# 157.71119337023583
# 171.48325
# 14.520281978411957
# 14.271794126988766
# 13.412561166114324
