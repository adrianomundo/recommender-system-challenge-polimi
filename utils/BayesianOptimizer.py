from bayes_opt import BayesianOptimization

from recommenders.Hybrids import Hybrid
from utils.data_handler import *
from utils.evaluation_functions import evaluate_algorithm


def run(als_weight, elastic_weight, item_cbf_weight, item_cf_weight, rp3_weight, slim_bpr_weight, user_cf_weight):

    urm_tuples = data_csv_splitter("urm")
    urm_all = urm_all_builder(urm_tuples)

    urm_train, urm_test = train_test_holdout(urm_all, 0.8)

    icm_asset_tuples = data_csv_splitter("icm_asset")
    icm_price_tuples = data_csv_splitter("icm_price")
    icm_sub_class_tuples = data_csv_splitter("icm_sub_class")
    icm_all = icm_all_builder(urm_all, icm_asset_tuples, icm_price_tuples, icm_sub_class_tuples)

    ucm_age_tuples = data_csv_splitter("ucm_age")
    ucm_region_tuples = data_csv_splitter("ucm_region")
    ucm_all = ucm_all_builder(urm_all, ucm_age_tuples, ucm_region_tuples)

    recommender = Hybrid.Hybrid(als_weight, elastic_weight, item_cbf_weight, item_cf_weight, rp3_weight,
                                slim_bpr_weight, user_cf_weight)
    recommender.fit(urm_train, icm_all, ucm_all, save_matrix=False, load_matrix=True)

    return evaluate_algorithm(urm_test, recommender)["MAP"]


if __name__ == '__main__':
    # Bounded region of parameter space
    pbounds = {'als_weight': (0.7, 0.95), 'elastic_weight': (2, 3.6), 'item_cbf_weight': (4.7, 5.8),
               'item_cf_weight': (4.8, 6.5), 'rp3_weight': (5.2, 6.2), 'slim_bpr_weight': (0, 0.04),
               'user_cf_weight': (0.05, 0.095)}

    optimizer = BayesianOptimization(
        f=run,
        pbounds=pbounds,
        verbose=2  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    optimizer.maximize(
        init_points=75,  # random steps
        n_iter=75
    )

    print(optimizer.max)
