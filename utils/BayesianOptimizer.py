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

    return evaluate_algorithm(urm_test, recommender)["recall"]


if __name__ == '__main__':
    # Bounded region of parameter space
    pbounds = {'als_weight': (0.65, 0.8), 'elastic_weight': (2.1, 3), 'item_cbf_weight': (5.3, 6.0),
               'item_cf_weight': (4.7, 5.5), 'rp3_weight': (5.2, 6.0), 'slim_bpr_weight': (0, 0.03),
               'user_cf_weight': (0.06, 0.09)}

    optimizer = BayesianOptimization(
        f=run,
        pbounds=pbounds,
        verbose=2  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    optimizer.maximize(
        init_points=75,  # random steps
        n_iter=75,
        kappa=0.1
    )

    print(optimizer.max)
