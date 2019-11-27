from utils.data_handler import *
from utils.evaluation_functions import *

urm_tuples = data_csv_splitter("urm")
urm_all = urm_all_builder(urm_tuples)


class RandomRecommender(object):

    def __init__(self):
        self.numItems = None

    def fit(self, urm_train):
        # shape[0] --> number of rows (# users), shape[1] --> number of columns (# items)
        self.numItems = urm_train.shape[1]

    # at=5 is the default value in case not furnished by the user
    def recommend(self, user_id, at=10):
        recommended_items = np.random.choice(self.numItems, at)

        return recommended_items


urm_train, urm_test = train_test_holdout(urm_all, 0.8)
# urm_train, urm_test = train_test_loo(urm_all)

randomRecommender = RandomRecommender()
randomRecommender.fit(urm_train)

print(evaluate_algorithm(urm_test, randomRecommender, 10))

target_list = target_list()
results = {}

for line in target_list:
    recommended_items = randomRecommender.recommend(line, 10)
    results[line] = recommended_items

# create_csv(results)
