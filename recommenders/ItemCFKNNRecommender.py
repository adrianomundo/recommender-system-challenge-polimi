from recommenders.TopPopRecommender import TopPopRecommender
from utils.evaluation_functions import *
from utils.compute_similarity_python import *
from utils.data_handler import *

urm_tuples = data_csv_splitter("urm")
urm_all = urm_all_builder(urm_tuples)

warm_users = get_warm_users(urm_all)


class ItemCFKNNRecommender(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, topK=50, shrink=100.0, normalize=True, similarity="tanimoto"):
        similarity_object = Compute_Similarity_Python(self.URM, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


urm_train, urm_test = train_test_holdout(urm_all, 0.8)
# urm_train, urm_test = train_test_loo(urm_all)

recommender = ItemCFKNNRecommender(urm_all)
recommender.fit(topK=10, shrink=50.0)

top_rec = TopPopRecommender()
top_rec.fit(urm_train)

print(evaluate_algorithm(urm_test, recommender, 10))

target_list = target_list()
results = {}

for line in target_list:
    if line in warm_users:
        recommended_items = recommender.recommend(line, 10)
        results[line] = recommended_items
    else:
        recommended_items = top_rec.recommend(line, 10)
        results[line] = recommended_items

# create_csv(results)
