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

    # Parameters related to itemCBF with hstack (useless for the moment)
    pbounds = {'als_weight': (0.65, 0.85), 'elastic_weight': (3.0, 3.3), 'item_cbf_weight': (0.7, 1.65),
               'item_cf_weight': (6.0, 6.4), 'rp3_weight': (5.7, 6.05), 'slim_bpr_weight': (0.07, 0.095),
               'user_cf_weight': (0.08, 0.11)}

    pbounds_new = {'als_weight': (0.6377, 0.8377), 'elastic_weight': (2.183, 2.383), 'item_cbf_weight': (5.77, 5.97),
                   'item_cf_weight': (5.084, 5.284), 'rp3_weight': (5.255, 5.455),
                   'slim_bpr_weight': (0.003048, 0.005048), 'user_cf_weight': (0.07906, 0.09906)}

    optimizer = BayesianOptimization(
        f=run,
        pbounds=pbounds_new,
        verbose=2  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    optimizer.maximize(
        init_points=50,  # random steps
        n_iter=100,
        xi=0.0
    )

    print(optimizer.max)
