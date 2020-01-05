import argparse

from recommenders.Base.RandomRecommender import RandomRecommender
from recommenders.Base.TopPopRecommender import TopPopRecommender
from recommenders.CBF.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from recommenders.CBF.UserCBFKNNRecommender import UserCBFKNNRecommender
from recommenders.CF.ItemCFKNNRecommender import ItemCFKNNRecommender
from recommenders.CF.UserCFKNNRecommender import UserCFKNNRecommender
from recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from recommenders.Hybrids.HybridRecommender import HybridRecommender
from recommenders.Hybrids.FallbackRecommender import FallbackRecommender
from recommenders.MF.ALS import ALSRecommender
from recommenders.MF.PureSVD import PureSVDRecommender
from recommenders.MF.SVD_ICM import SVD_ICM_Recommender
from recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from recommenders.SLIM_BPR.SLIM_BPR import SLIM_BPR
from recommenders.SLIM_ElasticNet.SLIM_ElasticNet import SLIMElasticNetRecommender
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

    def get_urm_all(self):
        urm_tuples = data_csv_splitter("urm")
        self.urm_all = urm_all_builder(urm_tuples)

    def get_icm_all(self, feature_weighting=False):
        icm_asset_tuples = data_csv_splitter("icm_asset")
        icm_price_tuples = data_csv_splitter("icm_price")
        icm_sub_class_tuples = data_csv_splitter("icm_sub_class")
        self.icm_all = icm_all_builder(self.urm_all, icm_asset_tuples, icm_price_tuples, icm_sub_class_tuples)
        if feature_weighting:
            self.icm_all = bm_25_feature_weighting(self.icm_all)

    def get_ucm_all(self, feature_weighting=False):
        ucm_age_tuples = data_csv_splitter("ucm_age")
        ucm_region_tuples = data_csv_splitter("ucm_region")
        self.ucm_all = ucm_all_builder(self.urm_all, ucm_age_tuples, ucm_region_tuples)
        if feature_weighting:
            self.ucm_all = bm_25_feature_weighting(self.ucm_all)

    def split_dataset_holdout(self):
        print("Splitting dataset using holdout function...")
        self.urm_train, self.urm_test = train_test_holdout(self.urm_all, 0.8)

    def split_dataset_loo(self):
        print("Splitting dataset using LeaveOneOut...")
        self.urm_train, self.urm_test = train_test_loo(self.urm_all)

    def fit_recommender(self):
        print("Fitting model...")
        self.get_urm_all()
        if self.evaluate:
            self.split_dataset_holdout()
            matrix = self.urm_train
        else:
            matrix = self.urm_all
        if self.name == 'random':
            self.recommender.fit(matrix)
        elif self.name == 'top_pop':
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
        elif self.name == 'RP3beta':
            self.recommender.fit(matrix)
        elif self.name == 'ALS':
            self.recommender.fit(matrix)
        elif self.name == 'PureSVD':
            self.recommender.fit(matrix)
        elif self.name == 'SVDICM':
            self.get_icm_all()
            self.recommender.fit(matrix, self.icm_all)
        elif self.name == 'hybrid':
            self.get_icm_all()
            self.get_ucm_all()
            self.recommender.fit(matrix, self.icm_all, self.ucm_all)
        elif self.name == 'fallback':
            self.get_ucm_all()
            self.recommender.fit(matrix, self.ucm_all)

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
    parser.add_argument('recommender', help="recommender type (required)", choices=['random', 'top_pop',
                                                                                    'itemCBF', 'userCBF',
                                                                                    'itemCF', 'userCF',
                                                                                    'SLIM_BPR', 'SLIM_BPR_Cython',
                                                                                    'SLIM_ElasticNet', 'RP3beta',
                                                                                    'ALS', 'PureSVD', 'SVDICM',
                                                                                    'hybrid', 'fallback'])
    parser.add_argument('--eval', help="enable evaluation", action="store_true")
    parser.add_argument('--csv', help="enable csv creation", action='store_true')
    args = parser.parse_args()

    recommender = None

    if args.recommender == 'random':
        print("random selected")
        recommender = RandomRecommender()

    elif args.recommender == 'top_pop':
        print("top_pop selected")
        recommender = TopPopRecommender()

    elif args.recommender == 'itemCBF':
        print("itemCBF selected")
        recommender = ItemCBFKNNRecommender()

    elif args.recommender == 'userCBF':
        print("userCBF selected")
        recommender = UserCBFKNNRecommender()

    elif args.recommender == 'itemCF':
        print("itemCF selected")
        recommender = ItemCFKNNRecommender()

    elif args.recommender == 'userCF':
        print("userCF selected")
        recommender = UserCFKNNRecommender()

    elif args.recommender == 'SLIM_BPR':
        print("SLIM_BPR selected")
        recommender = SLIM_BPR()

    elif args.recommender == 'SLIM_BPR_Cython':
        print("SLIM_BPR_Cython selected")
        recommender = SLIM_BPR_Cython()

    elif args.recommender == 'SLIM_ElasticNet':
        print("SLIM_ElasticNet selected")
        recommender = SLIMElasticNetRecommender()

    elif args.recommender == 'RP3beta':
        print("RP3beta selected")
        recommender = RP3betaRecommender()

    elif args.recommender == 'ALS':
        print("ALS selected")
        recommender = ALSRecommender()

    elif args.recommender == 'PureSVD':
        print("PureSVD selected")
        recommender = PureSVDRecommender()

    elif args.recommender == 'SVDICM':
        print("SVDICM selected")
        recommender = SVD_ICM_Recommender()

    elif args.recommender == 'hybrid':
        print("hybrid selected")
        recommender = HybridRecommender()

    elif args.recommender == 'fallback':
        print("fallback selected")
        recommender = FallbackRecommender()

    print(args)
    Runner(recommender, args.recommender, evaluate=args.eval, csv=args.csv).run()
