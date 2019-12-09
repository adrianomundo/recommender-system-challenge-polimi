import numpy as np

from recommenders.CBF import ItemCBFKNNRecommender
from recommenders.CF import ItemCFKNNRecommender, UserCFKNNRecommender
from recommenders.SLIM_BPR.Cython import SLIM_BPR_Cython
from recommenders.base import TopPopRecommender


class Hybrid(object):
    """ Hybrid
        Hybrid of four prediction scores R = R1*alpha + R2*beta + R3*gamma + R4*epsilon
    """

    def __init__(self):
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.epsilon = None
        self.item_cf_recommender = ItemCFKNNRecommender.ItemCFKNNRecommender()
        self.user_cf_recommender = UserCFKNNRecommender.UserCFKNNRecommender()
        self.item_cbf_recommender = ItemCBFKNNRecommender.ItemCBFKNNRecommender()
        self.slim_recommender = SLIM_BPR_Cython.SLIM_BPR_Cython()
        self.top_recommender = TopPopRecommender.TopPopRecommender()
        self.urm_train = None
        self.warm_users = None

    def fit(self, urm_train, warm_users, icm_all, alpha=1.478, beta=0.1759, gamma=1.458, epsilon=0.005797):

        self.urm_train = urm_train
        self.warm_users = warm_users
        self.item_cf_recommender.fit(urm_train)
        self.item_cbf_recommender.fit(urm_train, icm_all)
        self.user_cf_recommender.fit(urm_train)
        self.slim_recommender.fit(urm_train)
        self.top_recommender.fit(urm_train)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def compute_score(self, user_id):

        item_weights_1 = self.item_cf_recommender.compute_score(user_id)
        item_weights_2 = self.slim_recommender.compute_score(user_id)
        item_weights_3 = self.item_cbf_recommender.compute_score(user_id)
        item_weights_4 = self.user_cf_recommender.compute_score(user_id)

        item_weights = item_weights_1 * self.alpha
        item_weights += item_weights_2 * self.beta
        item_weights += item_weights_3 * self.gamma
        item_weights += item_weights_4 * self.epsilon

        return item_weights

    def filter_seen(self, user_id, scores):
        start_pos = self.urm_train.indptr[user_id]
        end_pos = self.urm_train.indptr[user_id + 1]

        user_profile = self.urm_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def recommend(self, user_id, at=10, exclude_seen=True):

        if user_id in self.warm_users:
            scores = self.compute_score(user_id)

            if exclude_seen:
                scores = self.filter_seen(user_id, scores)

            # rank items
            ranking = scores.argsort()[::-1]

            return ranking[:at]
        else:
            return self.top_recommender.recommend(user_id, at)
