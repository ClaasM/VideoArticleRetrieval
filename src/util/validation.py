import numpy as np
from scipy.spatial import distance


def ranking_validation(validation_data, model):
    x_validation = np.array([x for x, y in validation_data])
    y_validation = np.array([y for x, y in validation_data])
    y_predictions = model.predict(x_validation)
    # TODO is predictions/articles the right way around
    similarities = distance.cdist(y_predictions, y_validation, 'cosine')
    ranks = np.zeros(similarities.shape[0])
    for i in range(similarities.shape[0]):
        # Sort similarities, but keep indices not values
        indices_sorted = np.argsort(similarities[i])
        # The index of i is our rank.
        ranks[i] = np.where(indices_sorted == i)[0][0]  # TODO maybe np.int64(i)

    result = dict()
    result["r1"] = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    result["r5"] = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    result["r10"] = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    result["median_rank"] = np.floor(np.median(ranks)) + 1
    result["mean_rank"] = ranks.mean() + 1
    return result
