import argparse
from recommenders import RandomRecommender
from recommenders import TopPopRecommender
from utils.data_handler import *


class Runner:
    def __init__(self, recommender, name, evaluate=True):
        self.recommender = recommender
        self.name = name
        self.evaluate = evaluate


        self.urm_all = None

    def get_urm_all(self):
        urm_tuples = data_csv_splitter("urm")
        self.urm_all = urm_all_builder(urm_tuples)

    def get_dataset_split(self):
        urm_train, urm_test = train_test_holdout(self.urm_all, 0.8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recommender', choices=['random', 'top-pop'])
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--csv', action='store_true')
    args = parser.parse_args()

    recommender = None

    if args.recommender == 'random':
        print("random selected")
        recommender = RandomRecommender.RandomRecommender()

    elif args.recommender == 'top-pop':
        print("top-pop selected")
        recommender = TopPopRecommender.TopPopRecommender()

    print(args)
    Runner(recommender, args.recommender, evaluate=args.eval).run()
