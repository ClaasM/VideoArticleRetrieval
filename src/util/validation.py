import random

import numpy as np
from scipy.spatial import distance


def ranking_validation(validation_data, model):
    x_validation = np.array([x for x, y in validation_data])
    y_validation = np.array([y for x, y in validation_data])
    y_predictions = model.predict(x_validation)
    # text_embed_vecs = self.model.predict([np.array(text_vec_batch)])
    # TODO is predictions/articles the right way around
    similarities = distance.cdist(y_predictions, y_validation, 'cosine')
    index = random.randint(0, len(x_validation) - 1)
    print("Example prediction: %d->%d (should be %d)" %
          (sum(x_validation[index]), sum(y_predictions[index]), sum(y_validation[index])))
    ranks = np.zeros(similarities.shape[0])
    for i in range(similarities.shape[0]):
        # Sort similarities, but keep indices not values
        indices_sorted = np.argsort(similarities[0])
        # The index of i is our rank.
        ranks[i] = np.where(indices_sorted == i)[0][0]  # TODO maybe np.int64(i)

    # print(ranks)

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return r1, r5, r10, medr, meanr
