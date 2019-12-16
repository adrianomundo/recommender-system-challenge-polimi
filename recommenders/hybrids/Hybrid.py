import numpy as np

from recommenders.CBF import ItemCBFKNNRecommender
from recommenders.CF import ItemCFKNNRecommender, UserCFKNNRecommender
from recommenders.GraphBased import RP3betaRecommender
from recommenders.SLIM_ElasticNet import SLIM_ElasticNet
from recommenders.hybrids import UserCBFKNNTopPop


class Hybrid(object):

    def __init__(self, item_cbf_weight=2.069, item_cf_weight=1.056, elastic_weight=0.542, rp3_weight=1.404,
                 user_cf_weight=0.03977):

        self.urm_train = None

        self.item_cbf_weight = item_cbf_weight
        self.item_cf_weight = item_cf_weight
        self.elastic_weight = elastic_weight
        self.rp3_weight = rp3_weight
        self.user_cf_weight = user_cf_weight

        self.item_cbf_recommender = ItemCBFKNNRecommender.ItemCBFKNNRecommender()
        self.item_cf_recommender = ItemCFKNNRecommender.ItemCFKNNRecommender()
        self.elastic_recommender = SLIM_ElasticNet.SLIMElasticNetRecommender()
        self.rp3_recommender = RP3betaRecommender.RP3betaRecommender()
        self.user_cf_recommender = UserCFKNNRecommender.UserCFKNNRecommender()
        self.user_cbf_top_pop_recommender = UserCBFKNNTopPop.UserCBFKNNTopPop()

    def fit(self, urm_train, icm_all, ucm_all, load_matrix=False):

        self.urm_train = urm_train

        self.item_cbf_recommender.fit(urm_train, icm_all, load_matrix=load_matrix)
        self.item_cf_recommender.fit(urm_train, load_matrix=load_matrix)
        self.elastic_recommender.fit(urm_train, load_matrix=load_matrix)
        self.rp3_recommender.fit(urm_train, load_matrix=load_matrix)
        self.user_cf_recommender.fit(urm_train, load_matrix=load_matrix)
        self.user_cbf_top_pop_recommender.fit(urm_train, ucm_all, load_matrix=load_matrix)

    def compute_score(self, user_id):

        item_weights_1 = self.item_cbf_recommender.compute_score(user_id)
        item_weights_2 = self.item_cf_recommender.compute_score(user_id)
        item_weights_3 = self.elastic_recommender.compute_score(user_id)
        item_weights_4 = self.rp3_recommender.compute_score(user_id)
        item_weights_5 = self.user_cf_recommender.compute_score(user_id)

        item_weights = item_weights_1 * self.item_cbf_weight
        item_weights += item_weights_2 * self.item_cf_weight
        item_weights += item_weights_3 * self.elastic_weight
        item_weights += item_weights_4 * self.rp3_weight
        item_weights += item_weights_5 * self.user_cf_weight

        return item_weights

    def recommend(self, user_id, at=10, exclude_seen=True):

        scores = self.compute_score(user_id)

        if scores.sum() == 0.0:

            return self.user_cbf_top_pop_recommender.recommend(user_id, at)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = self.urm_train.indptr[user_id]
        end_pos = self.urm_train.indptr[user_id + 1]

        user_profile = self.urm_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores
