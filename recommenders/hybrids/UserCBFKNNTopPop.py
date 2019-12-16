import numpy as np

from recommenders.CBF import UserCBFKNNRecommender
from recommenders.base import TopPopRecommender


class UserCBFKNNTopPop(object):

    def __init__(self):

        self.urm_train = None
        self.ucm_all = None
        self.user_cbf_recommender = UserCBFKNNRecommender.UserCBFKNNRecommender()
        self.top_pop_recommender = TopPopRecommender.TopPopRecommender()

    def fit(self, urm_train, ucm_all, load_matrix=False):

        self.urm_train = urm_train
        self.ucm_all = ucm_all
        self.user_cbf_recommender.fit(self.urm_train, self.ucm_all, load_matrix=load_matrix)
        self.top_pop_recommender.fit(self.urm_train)

    def compute_score(self, user_id):

        return self.user_cbf_recommender.compute_score(user_id)

    def check_scores(self, scores, user_id, at):

        if scores.sum() == 0.0:

            return self.top_pop_recommender.recommend(user_id, at)

    def recommend(self, user_id, at=10, exclude_seen=True):

        scores = self.compute_score(user_id)

        self.check_scores(scores, user_id, at)

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
