from utils.data_handler import *
from utils.compute_similarity_python import *
from utils.evaluation_functions import *

urm_tuples = data_csv_splitter("urm")
urm_all = urm_all_builder(urm_tuples)

icm_sub_class = data_csv_splitter("icm_sub_class")
single_icm_builder(icm_sub_class)


class ItemCBFKNNRecommender(object):

    def __init__(self, urm, icm):
        self.urm = urm
        self.icm = icm
        self.W_sparse = None

    def fit(self, top_k=50, shrink=100.0, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.icm.T, shrink=shrink,
                                                      topK=top_k, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=10, exclude_seen=True):

        # compute the scores using the dot product
        # ravel(): returns a contiguous flattened array
        user_profile = self.urm[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id + 1]

        user_profile = self.urm.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


urm_train, urm_test = train_test_holdout(urm_all, 0.8)

recommender = ItemCBFKNNRecommender(urm_all, icm)
recommender.fit(top_k=50, shrink=0.0)

target_list = target_list()
results = {}

for line in target_list:
    recommended_items = recommender.recommend(line, 10)
    results[line] = recommended_items

create_csv(results)

print(evaluate_algorithm(urm_test, recommender))
