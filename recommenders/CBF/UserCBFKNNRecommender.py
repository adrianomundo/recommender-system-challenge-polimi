import numpy as np
from utils.compute_similarity_python import Compute_Similarity_Python


class UserCBFKNNRecommender(object):

    def __init__(self):
        self.urm_train = None
        self.ucm_all = None
        self.W_sparse = None

    def fit(self, urm_train, ucm_all, top_k=10, shrink=50.0, normalize=True, similarity="cosine"):

        self.urm_train = urm_train
        self.ucm_all = ucm_all

        similarity_object = Compute_Similarity_Python(self.ucm_all.T, shrink=shrink,
                                                      topK=top_k, normalize=normalize,
                                                      similarity=similarity)

        print("Computing similarity...")
        self.W_sparse = similarity_object.compute_similarity()

    def compute_score(self, user_id):

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
