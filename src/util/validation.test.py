from src.util.validation import ranking_validation
import numpy as np


class MockModel:
    def __init__(self, x_y_mapping):
        self.x_y_mapping = x_y_mapping

    def predict(self, data):
        results = list()
        for entry in data:
            if type(entry) != list:
                entry = entry.tolist()
            results.append(self.x_y_mapping[str(entry)])
        return np.array(results)


def test_ranking_validation():
    # Create mapping X-values -> y-predictions to mock a model
    model_mapping = {
        str([1]): [2]
    }
    # Data used as validation data (X-values -> y-truth)
    test_data = [
        ([1], [3])
    ]
    results = ranking_validation(test_data, MockModel(model_mapping))
    # results is r1, r5, r10, medr, meanr
    print(results)
    assert results == (100, 100, 100, 1, 1)

    # Now lets try a more complex example
    model_mapping = {
        str([1]): [1],
        str([2]): [2],
        str([3]): [3],
        str([4]): [4],
        str([5]): [5],
        str([6]): [6],
        str([7]): [7],
        str([8]): [8],
        str([9]): [9],
    }
    test_data = [
        ([1], [1.3]),
        ([2], [2.3]),
        ([3], [3.3]),
        ([4], [4.3]),
        ([5], [5.3]),
        ([6], [6.3]),
        ([7], [7.3]),
        ([8], [8.3]),
        ([9], [9.3]),
    ]
    results = ranking_validation(test_data, MockModel(model_mapping))
    # Should still be 100%, since the closest result is still always the right one
    print(results)
    assert results == (100, 100, 100, 1, 1)


if __name__ == '__main__':
    test_ranking_validation()
