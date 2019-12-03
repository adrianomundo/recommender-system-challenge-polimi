import numpy as np


class TopPopRecommender(object):

    def __init__(self):
        self.popularItems = None
        self.urm_all = None

    def fit(self, urm_all):

        self.urm_all = urm_all

        # (URM_train > 0) --> applies a filter to URM_train:
        # if an element of URM_train is > 0 --> True
        # if an element of URM_train is < 0 --> False
        # URM_train becomes a matrix of boolean values
        # sum(axis=0) --> sums all the item in each column
        item_popularity = (urm_all > 0).sum(axis=0)

        # squeeze to eliminate the extra dimension (dimension lost in the step above)
        item_popularity = np.array(item_popularity).squeeze()

        # argsort returns the indices that would sort an array
        # in this case the items' id
        self.popularItems = np.argsort(item_popularity)

        # flip inverts the array (from the most popular to the less popular)
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self, user_id, at=10, remove_seen=True):

        if remove_seen:

            # in1d --> tests whether each element of a 1-D array is also present in a second array
            # invert=True/False --> if True, the values in the returned array are inverted
            # (that is, False where an element of ar1 is in ar2 and True otherwise)
            unseen_items_mask = np.in1d(self.popularItems, self.urm_all[user_id].indices,
                                        assume_unique=True, invert=True)

            # applies the mask in order to eliminate the already seen elements
            unseen_items = self.popularItems[unseen_items_mask]

            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.popularItems[0:at]

        return recommended_items
