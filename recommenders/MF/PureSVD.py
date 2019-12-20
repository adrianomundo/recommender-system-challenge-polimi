from utils.data_handler import *
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sps
import numpy as np


class PureSVDRecommender(object):

    def __init__(self):

        self.urm_train = None
        self.num_factors = None
        self.USER_factors = None
        self.ITEM_factors = None
        self.random_seed = None
        self.U = None
        self.Sigma = None
        self.VT = None
        self.s_Vt = None

    def fit(self, urm_train, num_factors = 100, n_iter = 5, random_seed = None):

        print(" Computing PureSVD decomposition...")

        self.urm_train = urm_train

        self.U, self.Sigma, self.VT = randomized_svd(self.urm_train,
                                                n_components = num_factors,
                                                n_iter = n_iter,
                                                random_state = random_seed)

        self.s_Vt = sps.diags(self.Sigma)*self.VT

        self.USER_factors = self.U
        self.ITEM_factors = self.s_Vt.T

        print(" Computing PureSVD decomposition... Done!")

    def recommend(self, user_id, at=10, exclude_seen=True):

        scores = self.compute_score(user_id)

        # TODO undestand unseen_warm_items -> see repo

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def compute_score(self, user_id):

        user_profile = self.U[user_id]
        return user_profile.dot(self.s_Vt)

    def filter_seen(self, user_id, scores):

        start_pos = self.urm_train.indptr[user_id]
        end_pos = self.urm_train.indptr[user_id + 1]

        user_profile = self.urm_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


