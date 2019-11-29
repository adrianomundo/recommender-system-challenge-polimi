import argparse
from recommenders import RandomRecommender, TopPopRecommender, ItemCBFKNNRecommender, ItemCFKNNRecommender, \
    UserCFKNNRecommender
from recommenders.hybrids import ItemCFKNNTopPopHybrid, UserCFKNNTopPopHybrid
from utils.data_handler import *
from utils.evaluation_functions import evaluate_algorithm


class Runner:
    def __init__(self, recommender, name, evaluate=True, csv=False):
        self.recommender = recommender
        self.name = name
        self.evaluate = evaluate
        self.csv = csv

        self.urm_all = None
        self.urm_train = None
        self.urm_test = None

        self.icm_all = None
        self.ucm_all = None

        self.warm_users = None

    def get_urm_all(self):
        urm_tuples = data_csv_splitter("urm")
        self.urm_all = urm_all_builder(urm_tuples)

    def get_icm_all(self):
        icm_asset_tuples = data_csv_splitter("icm_asset")
        icm_price_tuples = data_csv_splitter("icm_price")
        icm_sub_class_tuples = data_csv_splitter("icm_sub_class")
        self.icm_all = icm_all_builder(self.urm_all, icm_asset_tuples, icm_price_tuples, icm_sub_class_tuples)

    def get_ucm_all(self):
        ucm_age_tuples = data_csv_splitter("ucm_age")
        ucm_region_tuples = data_csv_splitter("ucm_region")
        self.ucm_all = ucm_all_builder(self.urm_all, ucm_age_tuples, ucm_region_tuples)

    def split_dataset_holdout(self):
        print("Splitting dataset using holdout function...")
        self.urm_train, self.urm_test = train_test_holdout(self.urm_all, 0.8)

    def split_dataset_loo(self):
        print("Splitting dataset using LeaveOneOut...")
        self.urm_train, self.urm_test = train_test_loo(self.urm_all)

    def get_warm_users(self):
        self.warm_users = get_warm_users(self.urm_all)

    def fit_recommender(self):
        print("Fitting model...")
        self.get_urm_all()
        if not self.evaluate:
            if self.name == 'random' or self.name == 'top-pop':
                self.recommender.fit(self.urm_all)
            elif self.name == 'itemCBF':
                self.get_icm_all()
                self.recommender.fit(self.urm_all, self.icm_all, top_k=10, shrink=50.0)
            elif self.name == 'itemCF':
                self.recommender.fit(self.urm_all, top_k=10, shrink=50.0)
            elif self.name == 'userCF':
                self.recommender.fit(self.urm_all, top_k=500, shrink=300.0)
            elif self.name == 'hybrid':
                self.get_warm_users()
                self.recommender.fit(self.urm_all, self.warm_users)
        else:
            self.split_dataset_holdout()
            if self.name == 'random' or self.name == 'top-pop':
                self.recommender.fit(self.urm_train)
            elif self.name == 'itemCBF':
                self.get_icm_all()
                self.recommender.fit(self.urm_train, self.icm_all, top_k=10, shrink=50.0)
            elif self.name == 'itemCF':
                self.recommender.fit(self.urm_train, top_k=10, shrink=50.0)
            elif self.name == 'userCF':
                self.recommender.fit(self.urm_train, top_k=500, shrink=300.0)
            elif self.name == 'hybrid':
                self.get_warm_users()
                self.recommender.fit(self.urm_train, self.warm_users)
        print("Model fitted")

    def run_recommendations(self):
        target_users = target_users_list()
        results = {}
        print("Computing recommendations...")
        for user in tqdm(target_users):
            recommended_items = self.recommender.recommend(user, 10)
            results[user] = recommended_items

        if self.evaluate:
            print("Evaluating...")
            evaluate_algorithm(self.urm_test, self.recommender, 10)

        if self.csv:
            print("Creating CSV file...")
            create_csv(results)

    def run(self):
        self.fit_recommender()
        self.run_recommendations()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recommender', help="recommender type (required)", choices=['random', 'top-pop',
                                                                                    'itemCBF', 'itemCF', 'userCF',
                                                                                    'hybrid'])
    parser.add_argument('--eval', help="enable evaluation", action="store_true")
    parser.add_argument('--csv', help="enable csv creation", action='store_true')
    args = parser.parse_args()

    recommender = None

    if args.recommender == 'random':
        print("random selected")
        recommender = RandomRecommender.RandomRecommender()

    elif args.recommender == 'top-pop':
        print("top-pop selected")
        recommender = TopPopRecommender.TopPopRecommender()

    elif args.recommender == 'itemCBF':
        print("itemCBF selected")
        recommender = ItemCBFKNNRecommender.ItemCBFKNNRecommender()

    elif args.recommender == 'itemCF':
        print("itemCF selected")
        recommender = ItemCFKNNRecommender.ItemCFKNNRecommender()

    elif args.recommender == 'userCF':
        print("userCF selected")
        recommender = UserCFKNNRecommender.UserCFKNNRecommender()

    elif args.recommender == 'hybrid':
        print("hybrid selected")
        recommender = ItemCFKNNTopPopHybrid.ItemCFKNNTopPopHybrid()

    print(args)
    Runner(recommender, args.recommender, evaluate=args.eval, csv=args.csv).run()
