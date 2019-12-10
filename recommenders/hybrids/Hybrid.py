import numpy as np

from recommenders.CBF import ItemCBFKNNRecommender
from recommenders.CF import ItemCFKNNRecommender, UserCFKNNRecommender
from recommenders.SLIM_BPR.Cython import SLIM_BPR_Cython
from recommenders.SLIM_ElasticNet import SLIM_ElasticNet
from recommenders.base import TopPopRecommender


class Hybrid(object):

    def __init__(self, item_cf_weight=1.154, slim_weight=0.207, item_cbf_weight=1.621,
                 user_cf_weight=0.01389, elastic_weight=0.2):

        self.item_cf_weight = item_cf_weight
        self.slim_weight = slim_weight
        self.item_cbf_weight = item_cbf_weight
        self.user_cf_weight = user_cf_weight
        self.elastic_weight = elastic_weight
        self.item_cf_recommender = ItemCFKNNRecommender.ItemCFKNNRecommender()
        self.slim_recommender = SLIM_BPR_Cython.SLIM_BPR_Cython()
        self.item_cbf_recommender = ItemCBFKNNRecommender.ItemCBFKNNRecommender()
        self.user_cf_recommender = UserCFKNNRecommender.UserCFKNNRecommender()
        self.elastic_recommender = SLIM_ElasticNet.SLIMElasticNetRecommender()
        self.top_recommender = TopPopRecommender.TopPopRecommender()
        self.urm_train = None
        self.warm_users = None

    def fit(self, urm_train, warm_users, icm_all):

        self.urm_train = urm_train
        self.warm_users = warm_users
        self.item_cf_recommender.fit(urm_train)
        self.item_cbf_recommender.fit(urm_train, icm_all)
        self.user_cf_recommender.fit(urm_train)
        self.slim_recommender.fit(urm_train)
        self.elastic_recommender.fit(urm_train)
        self.top_recommender.fit(urm_train)

    def compute_score(self, user_id):

        item_weights_1 = self.item_cf_recommender.compute_score(user_id)
        item_weights_2 = self.slim_recommender.compute_score(user_id)
        item_weights_3 = self.item_cbf_recommender.compute_score(user_id)
        item_weights_4 = self.user_cf_recommender.compute_score(user_id)
        item_weights_5 = self.elastic_recommender.compute_score(user_id)

        item_weights = item_weights_1 * self.item_cf_weight
        item_weights += item_weights_2 * self.slim_weight
        item_weights += item_weights_3 * self.item_cbf_weight
        item_weights += item_weights_4 * self.user_cf_weight
        item_weights += item_weights_5 * self.elastic_weight

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
