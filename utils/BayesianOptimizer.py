from bayes_opt import BayesianOptimization

from recommenders.Hybrids.HybridRecommender import HybridRecommender
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

    recommender = HybridRecommender(als_weight, elastic_weight, item_cbf_weight, item_cf_weight, rp3_weight,
                                    slim_bpr_weight, user_cf_weight)
    recommender.fit(urm_train, icm_all, ucm_all, save_matrix=False, load_matrix=True)

    return evaluate_algorithm(urm_test, recommender)["MAP"]


if __name__ == '__main__':
    # Bounded region of parameter space
    pbounds_old = {'als_weight': (1, 2), 'elastic_weight': (5.5, 7), 'item_cbf_weight': (4.5, 6),
                   'item_cf_weight': (6.5, 8), 'rp3_weight': (5.5, 7), 'slim_bpr_weight': (2.5, 3.5),
                   'user_cf_weight': (0, 0.04)}

    pbounds_new = {'als_weight': (0, 1), 'elastic_weight': (1, 3), 'item_cbf_weight': (4.5, 6.5),
                   'item_cf_weight': (4.5, 6), 'rp3_weight': (4.8, 6.5), 'slim_bpr_weight': (0, 1),
                   'user_cf_weight': (0, 1)}

    optimizer = BayesianOptimization(
        f=run,
        pbounds=pbounds_new,
        verbose=2  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    optimizer.maximize(
        init_points=50,  # random steps
        n_iter=100,
    )

    print(optimizer.max)
