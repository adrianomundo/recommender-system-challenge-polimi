import numpy as np
import scipy.sparse as sps
from utils.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython


class UserCFKNNRecommender(object):

    def __init__(self):
        self.urm_train = None
        self.W_sparse = None

    def fit(self, urm_train, top_k=590, shrink=0.0, normalize=True, similarity="cosine"):

        self.urm_train = urm_train

        similarity_object = Compute_Similarity_Cython(self.urm_train.T, shrink=shrink,
                                                      topK=top_k, normalize=normalize,
                                                      similarity=similarity)

        print("Computing userCF similarity...")
        self.W_sparse = similarity_object.compute_similarity()

        sps.save_npz("../tmp/userCF_matrix.npz", self.W_sparse)

    def compute_score(self, user_id, urm_train=None, load_matrix=True):

        if load_matrix:
            self.urm_train = urm_train
            self.W_sparse = sps.load_npz("../tmp/userCF_matrix.npz")

        # compute the scores using the dot product
        return self.W_sparse[user_id, :].dot(self.urm_train).toarray().ravel()

    def recommend(self, user_id, at=10, exclude_seen=True):

        scores = self.compute_score(user_id)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm_train.indptr[user_id]
        end_pos = self.urm_train.indptr[user_id + 1]

        user_profile = self.urm_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores
