from utils.data_handler import *

urm_tuples = data_train_csv_splitter()
urm_all = urm_builder(urm_tuples)


class RandomRecommender(object):

    def fit(self, urm_train):
        # shape[0] --> number of rows (# users), shape[1] --> number of columns (# items)
        self.numItems = urm_train.shape[1]

    # at=5 is the default value in case not furnished by the user
    def recommend(self, user_id, at=5):
        recommended_items = np.random.choice(self.numItems, at)

        return recommended_items


# urm_all split with RandomRecommender not needed
urm_train, urm_test = train_test_holdout(urm_all, 0.8)

randomRecommender = RandomRecommender()
randomRecommender.fit(urm_all)

target_list = target_list()
results = {}

for line in target_list:
    recommended_items = randomRecommender.recommend(line, 10)
    results[line] = recommended_items

create_csv(results)
