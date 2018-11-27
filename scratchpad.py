import pickle
import zlib

import numpy as np

test_list = list()

for i in range(2048):
    if i % 3 == 0:
        test_list.append(3)
    else:
        test_list.append(0)

test_sum = sum(test_list)

test_numpy = np.array(test_list)

string = str(test_numpy)
print("pickle.dumps: %d" % len(bytes(pickle.dumps(test_numpy))))
print("bytes: %d" % len(bytes(test_numpy)))
print("str: %d" % len(bytes(str(test_numpy), "utf-8")))
print("zlib_numpy: %d" % len(zlib.compress(test_numpy, 9)))
print("zlib: %d" % len(zlib.compress(test_list, 9)))

# utf-8: 203 chars
# pickle.dumps: 16543
