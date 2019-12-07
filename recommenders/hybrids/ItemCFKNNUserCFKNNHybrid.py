import numpy as np

from recommenders.CF import ItemCFKNNRecommender, UserCFKNNRecommender
from recommenders.base import TopPopRecommender


class ItemCFKNNUserCFKNNHybrid(object):
    """ ItemCFKNNUserCFKNNHybrid
        Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    """

    def __init__(self):
        self.alpha = None
        self.item_cf_recommender = ItemCFKNNRecommender.ItemCFKNNRecommender()
        self.user_cf_recommender = UserCFKNNRecommender.UserCFKNNRecommender()
        self.top_recommender = TopPopRecommender.TopPopRecommender()
        self.urm_train = None
        self.warm_users = None

    def fit(self, urm_train, warm_users, alpha=0.8):

        self.urm_train = urm_train
        self.warm_users = warm_users
        self.item_cf_recommender.fit(urm_train)
        self.user_cf_recommender.fit(urm_train)
        self.top_recommender.fit(urm_train)
        self.alpha = alpha

    def compute_score(self, user_id):

        item_weights_1 = self.item_cf_recommender.compute_score(user_id)
        item_weights_2 = self.user_cf_recommender.compute_score(user_id)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)

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
