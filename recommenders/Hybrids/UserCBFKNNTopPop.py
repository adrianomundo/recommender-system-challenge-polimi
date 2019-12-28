import numpy as np

from recommenders.Base import TopPopRecommender
from recommenders.CBF import UserCBFKNNRecommender


class UserCBFKNNTopPop(object):

    def __init__(self):

        self.urm_train = None
        self.ucm_all = None
        self.user_cbf_recommender = UserCBFKNNRecommender.UserCBFKNNRecommender()
        self.top_pop_recommender = TopPopRecommender.TopPopRecommender()

    def fit(self, urm_train, ucm_all, save_matrix=False, load_matrix=False):

        self.urm_train = urm_train
        self.ucm_all = ucm_all

        self.user_cbf_recommender.fit(self.urm_train, self.ucm_all, save_matrix=save_matrix, load_matrix=load_matrix)
        self.top_pop_recommender.fit(self.urm_train)

    def compute_score(self, user_id):

        return self.user_cbf_recommender.compute_score(user_id)

    def recommend(self, user_id, at=10, exclude_seen=True):

        scores = self.compute_score(user_id)

        if scores.sum() == 0.0:

            return self.top_pop_recommender.recommend(user_id, at)

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

    ''' 
    # version with merged ranking
    def recommend(self, user_id, at=10):

    user_cbf_score = self.user_cbf_recommender.compute_score(user_id)
    top_pop_ranking = self.top_pop_recommender.recommend(user_id, at)

    if user_cbf_score.sum() == 0.0:
        return top_pop_ranking

    user_cbf_ranking = self.user_cbf_recommender.recommend(user_id, at)

    intersection = np.intersect1d(user_cbf_ranking, top_pop_ranking)

    if len(intersection) == 0:
        return user_cbf_score

    merged_ranking = {}
    indices_user_cbf = []
    indices_top_pop = []

    for item in intersection:

        index_1 = np.where(user_cbf_ranking == item)
        indices_user_cbf.append(index_1)
        index_2 = np.where(top_pop_ranking == item)
        indices_top_pop.append(index_2)
        tmp_array = np.array([index_1, index_2])

        median = np.median(tmp_array)

        merged_ranking[item] = median

    merged_ranking = sorted(merged_ranking, key=merged_ranking.__getitem__)
    merged_ranking = np.asarray(merged_ranking)

    new_user_cbf_ranking = np.delete(user_cbf_ranking, indices_user_cbf)
    new_top_pop_ranking = np.delete(top_pop_ranking, indices_top_pop)

    merged_ranking = np.append(merged_ranking, new_user_cbf_ranking)

    return merged_ranking
    '''
