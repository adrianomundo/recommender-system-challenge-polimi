import argparse
from recommenders.SLIM_ElasticNet import SLIM_ElasticNet
from recommenders.SLIM_BPR import SLIM_BPR
from recommenders.SLIM_BPR.Cython import SLIM_BPR_Cython
from recommenders.base import RandomRecommender, TopPopRecommender
from recommenders.CBF import UserCBFKNNRecommender, ItemCBFKNNRecommender
from recommenders.CF import ItemCFKNNRecommender, UserCFKNNRecommender
from recommenders.hybrids import Hybrid
from utils.data_handler import *
from utils.evaluation_functions import evaluate_algorithm


class Runner:
    def __init__(self, recommender_object, name, evaluate=True, csv=False):
        self.recommender = recommender_object
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
        if self.evaluate:
            self.split_dataset_holdout()
            matrix = self.urm_train
        else:
            matrix = self.urm_all
        if self.name == 'random' or self.name == 'top-pop':
            self.recommender.fit(matrix)
        elif self.name == 'itemCBF':
            self.get_icm_all()
            self.recommender.fit(matrix, self.icm_all)
        elif self.name == 'userCBF':
            self.get_ucm_all()
            self.recommender.fit(matrix, self.ucm_all)
        elif self.name == 'itemCF':
            self.recommender.fit(matrix)
        elif self.name == 'userCF':
            self.recommender.fit(matrix)
        elif self.name == 'SLIM_BPR':
            self.recommender.fit(matrix)
        elif self.name == 'SLIM_BPR_Cython':
            self.recommender.fit(matrix)
        elif self.name == 'SLIM_ElasticNet':
            self.recommender.fit(matrix)
        elif self.name == 'hybrid':
            self.get_warm_users()
            self.get_icm_all()
            self.recommender.fit(matrix, self.warm_users, self.icm_all)
        print("Model fitted")

    def run_recommendations(self):
        if self.evaluate:
            print("Evaluating...")
            evaluate_algorithm(self.urm_test, self.recommender)

        if self.csv:
            target_users = target_users_list()
            results = {}
            print("Computing recommendations...")
            for user in tqdm(target_users):
                recommended_items = self.recommender.recommend(user)
                results[user] = recommended_items
            print("Creating CSV file...")
            create_csv(results)
            print("CSV file created")

    def run(self):
        self.fit_recommender()
        self.run_recommendations()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recommender', help="recommender type (required)", choices=['random', 'top-pop',
                                                                                    'itemCBF', 'userCBF',
                                                                                    'itemCF', 'userCF',
                                                                                    'SLIM_BPR', 'SLIM_BPR_Cython',
                                                                                    'SLIM_ElasticNet',
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

    elif args.recommender == 'userCBF':
        print("userCBF selected")
        recommender = UserCBFKNNRecommender.UserCBFKNNRecommender()

    elif args.recommender == 'itemCF':
        print("itemCF selected")
        recommender = ItemCFKNNRecommender.ItemCFKNNRecommender()

    elif args.recommender == 'userCF':
        print("userCF selected")
        recommender = UserCFKNNRecommender.UserCFKNNRecommender()

    elif args.recommender == 'SLIM_BPR':
        print("SLIM_BPR selected")
        recommender = SLIM_BPR.SLIM_BPR()

    elif args.recommender == 'SLIM_BPR_Cython':
        print("SLIM_BPR_Cython selected")
        recommender = SLIM_BPR_Cython.SLIM_BPR_Cython()

    elif args.recommender == 'SLIM_ElasticNet':
        print("SLIM_ElasticNet selected")
        recommender = SLIM_ElasticNet.SLIMElasticNetRecommender()

    elif args.recommender == 'hybrid':
        print("hybrid selected")
        recommender = Hybrid.Hybrid()

    print(args)
    Runner(recommender, args.recommender, evaluate=args.eval, csv=args.csv).run()
