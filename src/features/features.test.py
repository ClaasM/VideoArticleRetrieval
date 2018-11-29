"""
Some feature exploration/validating that everything works correctly
"""
from src.features.common import get_data

import numpy as np


def test():
    data, validation_data = get_data()

    x = np.array([x for x, y in data])
    y = np.array([y for x, y in data])
    x_validation = np.array([x for x, y in validation_data])
    y_validation = np.array([y for x, y in validation_data])

    print(np.sum(np.mean(x, axis=0)))
    print(np.sum(np.mean(y, axis=0)))
    print(np.sum(np.mean(x_validation, axis=0)))
    print(np.sum(np.mean(y_validation, axis=0)))


"""
13.780939217594032
165.97186
13.937101405743872
161.55359
"""

if __name__ == '__main__':
    test()
