import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

"""
Recommender with SVD: Singular Value Decomposition technique applied to the item content matrix. 
"""


class SVD_ICM_Recommender(object):

    def __init__(self, n_factors=2000, k=100):

        self.n_factors = n_factors
        self.k = k
        self.urm_train = None
        self.icm_all = None
        self.S_ICM_SVD = None

    def fit(self, urm_train, icm_all):

        self.urm_train = urm_train
        self.icm_all = icm_all

        self.S_ICM_SVD = self.get_S_ICM_SVD(self.icm_all, n_factors=self.n_factors)

    def get_S_ICM_SVD(self, icm_all, n_factors):

        print("Computing S_ICM_SVD...")

        self.icm_all = self.icm_all.astype(np.float64)
        self.icm_all = sparse.csr_matrix(self.icm_all)
        u, s, vt = svds(icm_all, k=n_factors, which='LM')

        ut = u.T

        s_2_flatten = np.power(s, 2)
        s_2 = np.diagflat(s_2_flatten)
        s_2_csr = sparse.csr_matrix(s_2)

        S = u.dot(s_2_csr.dot(ut))

        return S

    def compute_score(self, user_id):

        user_profile = self.urm_train[user_id]
        ratings = user_profile.dot(self.S_ICM_SVD)
        return ratings[0]

    def recommend(self, user_id, at=10, exclude_seen=True):

        scores = self.compute_score(user_id)

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
