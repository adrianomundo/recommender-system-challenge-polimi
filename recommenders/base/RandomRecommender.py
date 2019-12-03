import numpy as np


class RandomRecommender(object):

    def __init__(self):
        self.numItems = None

    def fit(self, urm_train):
        # shape[0] --> number of rows (# users), shape[1] --> number of columns (# items)
        self.numItems = urm_train.shape[1]

    # at=10 is the default value in case not furnished by the user
    def recommend(self, user_id, at=10):
        recommended_items = np.random.choice(self.numItems, at)

        return recommended_items
