import numpy as np
import scipy.sparse as sps
from utils.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython

from recommenders.base import TopPopRecommender


class UserCBFKNNRecommender(object):

    def __init__(self):
        self.urm_train = None
        self.ucm_all = None
        self.W_sparse = None
        self.top_rec = TopPopRecommender.TopPopRecommender()

    def fit(self, urm_train, ucm_all, top_k=800, shrink=5.0, normalize=True, similarity="cosine", load_matrix=False):

        self.urm_train = urm_train
        self.top_rec.fit(self.urm_train)

        if not load_matrix:
            print("Computing userCBF similarity...")
            self.ucm_all = ucm_all
            similarity_object = Compute_Similarity_Cython(self.ucm_all.T, shrink=shrink,
                                                          topK=top_k, normalize=normalize,
                                                          similarity=similarity)

            self.W_sparse = similarity_object.compute_similarity()
            sps.save_npz("../tmp/userCBF_similarity_matrix.npz", self.W_sparse)
        else:
            print("Loading userCBF_similarity_matrix.npz file...")
            self.W_sparse = sps.load_npz("../tmp/userCBF_similarity_matrix.npz")
            print("Matrix loaded!")

    def compute_score(self, user_id):

        tmp = self.W_sparse[user_id, :]
        return tmp.dot(self.urm_train).toarray().ravel()

    def recommend(self, user_id, at=10, exclude_seen=True):

        scores = self.compute_score(user_id)

        if scores.sum() == 0.0:

            return self.top_rec.recommend(user_id, at)

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
