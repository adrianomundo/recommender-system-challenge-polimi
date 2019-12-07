from bayes_opt import BayesianOptimization

from recommenders.CF import ItemCFKNNRecommender
from utils.data_handler import *
from utils.evaluation_functions import evaluate_algorithm


def item_cf(top_k, shrink):
    recommender = ItemCFKNNRecommender.ItemCFKNNRecommender()

    urm_tuples = data_csv_splitter("urm")
    urm_all = urm_all_builder(urm_tuples)

    urm_train, urm_test = train_test_holdout(urm_all, 0.8)

    recommender.fit(urm_train, top_k=int(top_k), shrink=shrink)

    return evaluate_algorithm(urm_test, recommender)["MAP"]


if __name__ == '__main__':
    # Bounded region of parameter space
    pbounds = {'top_k': (9, 11), 'shrink': (25, 30)}

    optimizer = BayesianOptimization(
        f=item_cf,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,  # random steps
        n_iter=50,
    )

    print(optimizer.max)
