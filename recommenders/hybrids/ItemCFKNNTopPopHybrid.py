from recommenders import ItemCFKNNRecommender, TopPopRecommender


class ItemCFKNNTopPopHybrid(object):

    def __init__(self):
        self.cf_recommender = ItemCFKNNRecommender.ItemCFKNNRecommender()
        self.top_recommender = TopPopRecommender.TopPopRecommender()
        self.warm_users = None

    def fit(self, urm_train, warm_users):
        self.cf_recommender.fit(urm_train)
        self.top_recommender.fit(urm_train)
        self.warm_users = warm_users

    def recommend(self, user_id, at=10):
        if user_id in self.warm_users:
            recommended_items = self.cf_recommender.recommend(user_id, at)
        else:
            recommended_items = self.top_recommender.recommend(user_id, at)

        return recommended_items
