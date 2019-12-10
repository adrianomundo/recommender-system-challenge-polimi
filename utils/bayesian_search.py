from bayes_opt import BayesianOptimization

from recommenders.hybrids import Hybrid
from utils.data_handler import *
from utils.evaluation_functions import evaluate_algorithm


def run(item_cf_weight, slim_weight, item_cbf_weight, user_cf_weight, elastic_weight):

    urm_tuples = data_csv_splitter("urm")
    urm_all = urm_all_builder(urm_tuples)

    urm_train, urm_test = train_test_holdout(urm_all, 0.8)

    icm_asset_tuples = data_csv_splitter("icm_asset")
    icm_price_tuples = data_csv_splitter("icm_price")
    icm_sub_class_tuples = data_csv_splitter("icm_sub_class")
    icm_all = icm_all_builder(urm_all, icm_asset_tuples, icm_price_tuples, icm_sub_class_tuples)

    warm_users = get_warm_users(urm_all)

    recommender = Hybrid.Hybrid(item_cf_weight, slim_weight, item_cbf_weight, user_cf_weight, elastic_weight)
    recommender.fit(urm_train, warm_users, icm_all)

    return evaluate_algorithm(urm_test, recommender)["MAP"]


if __name__ == '__main__':
    # Bounded region of parameter space
    pbounds = {'item_cf_weight': (1, 2.5), 'slim_weight': (0, 0.5), 'item_cbf_weight': (1, 2.5),
               'user_cf_weight': (0, 0.02), 'elastic_weight': (0, 1)}

    optimizer = BayesianOptimization(
        f=run,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    optimizer.maximize(
        init_points=30,  # random steps
        n_iter=120,
    )

    print(optimizer.max)
