import numpy as np
import scipy.sparse as sps
from utils.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython


class ItemCFKNNRecommender(object):

    def __init__(self):
        self.urm_train = None
        self.W_sparse = None

    def fit(self, urm_train, top_k=10, shrink=30.0, normalize=True, similarity="jaccard",
            save_matrix=False, load_matrix=False):

        self.urm_train = urm_train

        if not load_matrix:
            print("Computing itemCF similarity...")
            similarity_object = Compute_Similarity_Cython(self.urm_train, shrink=shrink,
                                                          topK=top_k, normalize=normalize,
                                                          similarity=similarity)

            self.W_sparse = similarity_object.compute_similarity()
            if save_matrix:
                sps.save_npz("../tmp/itemCF_similarity_matrix.npz", self.W_sparse)
                print("Matrix saved!")
        else:
            print("Loading itemCF_similarity_matrix.npz file...")
            self.W_sparse = sps.load_npz("../tmp/itemCF_similarity_matrix.npz")
            print("Matrix loaded!")

    def compute_score(self, user_id):

        user_profile = self.urm_train[user_id]

        return user_profile.dot(self.W_sparse).toarray().ravel()

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
