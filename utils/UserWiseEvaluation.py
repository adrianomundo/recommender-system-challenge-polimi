from matplotlib import pyplot

from recommenders.CBF import ItemCBFKNNRecommender
from recommenders.CF import ItemCFKNNRecommender, UserCFKNNRecommender
from recommenders.GraphBased import RP3betaRecommender
from recommenders.SLIM_BPR.Cython import SLIM_BPR_Cython
from recommenders.SLIM_ElasticNet import SLIM_ElasticNet
from recommenders.base import TopPopRecommender
from recommenders.Hybrids import UserCBFKNNTopPop, Hybrid
from utils.data_handler import *
from utils.evaluation_functions import user_wise_evaluation


class UserWiseEvaluation(object):

    def __init__(self):

        self.urm_train = None

        self.elastic_recommender = SLIM_ElasticNet.SLIMElasticNetRecommender()
        self.item_cbf_recommender = ItemCBFKNNRecommender.ItemCBFKNNRecommender()
        self.item_cf_recommender = ItemCFKNNRecommender.ItemCFKNNRecommender()
        self.rp3_recommender = RP3betaRecommender.RP3betaRecommender()
        self.slim_bpr_recommender = SLIM_BPR_Cython.SLIM_BPR_Cython()
        self.top_pop_recommender = TopPopRecommender.TopPopRecommender()
        self.user_cf_recommender = UserCFKNNRecommender.UserCFKNNRecommender()

        self.fallback_recommender = UserCBFKNNTopPop.UserCBFKNNTopPop()
        self.fallback_with_hstack_recommender = UserCBFKNNTopPop.UserCBFKNNTopPop()

        self.hybrid_recommender = Hybrid.Hybrid()

    def fit(self, urm_train, icm_all, ucm_all, load_matrix=False):

        self.urm_train = urm_train

        self.elastic_recommender.fit(urm_train, load_matrix=load_matrix)
        self.item_cbf_recommender.fit(urm_train, icm_all, load_matrix=load_matrix)
        self.item_cf_recommender.fit(urm_train, load_matrix=load_matrix)
        self.rp3_recommender.fit(urm_train, load_matrix=load_matrix)
        self.slim_bpr_recommender.fit(urm_train, load_matrix=load_matrix)
        self.top_pop_recommender.fit(urm_train)
        self.user_cf_recommender.fit(urm_train, load_matrix=load_matrix)

        self.fallback_recommender.fit(urm_train, ucm_all, load_matrix=load_matrix)
        self.fallback_with_hstack_recommender.fit(urm_train, hstack((self.urm_train, ucm_all)),
                                                  load_matrix=load_matrix)

        self.hybrid_recommender.fit(urm_train, icm_all, ucm_all, load_matrix=load_matrix)
        # self.hybrid_recommender.fit(urm_train, icm_all, hstack((self.urm_train, ucm_all)), load_matrix=load_matrix)

    def evaluate_and_plot(self, urm_test, initial_target_users):

        map_elastic_per_group = []
        map_item_cbf_per_group = []
        map_item_cf_per_group = []
        map_rp3_per_group = []
        map_slim_bpr_per_group = []
        map_top_pop_per_group = []
        map_user_cf_per_group = []

        map_fallback_per_group = []
        map_fallback_with_hstack_per_group = []

        map_hybrid_per_group = []

        profile_length = np.ediff1d(self.urm_train.indptr)
        block_size = int(len(profile_length) * 0.05)
        sorted_users = np.argsort(profile_length)

        total_number_of_users = 0

        for group_id in range(0, 20):

            start_pos = group_id * block_size
            end_pos = min((group_id + 1) * block_size, len(profile_length))

            users_in_group = sorted_users[start_pos:end_pos]
            total_number_of_users += len(users_in_group)
            users_in_group_p_len = profile_length[users_in_group]

            print("Group {}, average p.len {:.4f}, min {}, max {}".format(group_id, users_in_group_p_len.mean(),
                                                                          users_in_group_p_len.min(),
                                                                          users_in_group_p_len.max()))

            users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
            users_not_in_group = sorted_users[users_not_in_group_flag]

            users_in_group = list(set(initial_target_users) - set(list(users_not_in_group)))

            print("Evaluating ElasticNet...")
            results = user_wise_evaluation(urm_test, users_in_group, self.elastic_recommender, at=10)
            map_elastic_per_group.append(results)

            print("Evaluating itemCBF...")
            results = user_wise_evaluation(urm_test, users_in_group, self.item_cbf_recommender, at=10)
            map_item_cbf_per_group.append(results)

            print("Evaluating itemCF...")
            results = user_wise_evaluation(urm_test, users_in_group, self.item_cf_recommender, at=10)
            map_item_cf_per_group.append(results)

            print("Evaluating RP3beta...")
            results = user_wise_evaluation(urm_test, users_in_group, self.rp3_recommender, at=10)
            map_rp3_per_group.append(results)

            print("Evaluating SLIM_BPR...")
            results = user_wise_evaluation(urm_test, users_in_group, self.slim_bpr_recommender, at=10)
            map_slim_bpr_per_group.append(results)

            print("Evaluating top-pop...")
            results = user_wise_evaluation(urm_test, users_in_group, self.top_pop_recommender, at=10)
            map_top_pop_per_group.append(results)

            print("Evaluating userCF...")
            results = user_wise_evaluation(urm_test, users_in_group, self.user_cf_recommender, at=10)
            map_user_cf_per_group.append(results)

            print("Evaluating fallback...")
            results = user_wise_evaluation(urm_test, users_in_group, self.fallback_recommender, at=10)
            map_fallback_per_group.append(results)

            print("Evaluating fallback (with hstack)...")
            results = user_wise_evaluation(urm_test, users_in_group, self.fallback_with_hstack_recommender, at=10)
            map_fallback_with_hstack_per_group.append(results)

            print("Evaluating hybrid...")
            results = user_wise_evaluation(urm_test, users_in_group, self.hybrid_recommender, at=10)
            map_hybrid_per_group.append(results)

        print("Total number of users evaluated: " + str(total_number_of_users))

        pyplot.plot(map_elastic_per_group, label="ElasticNet")
        pyplot.plot(map_item_cbf_per_group, label="itemCBF")
        pyplot.plot(map_item_cf_per_group, label="itemCF")
        pyplot.plot(map_rp3_per_group, label="RP3beta")
        pyplot.plot(map_slim_bpr_per_group, label="SLIM_BPR")
        pyplot.plot(map_top_pop_per_group, label="Top-Pop")
        pyplot.plot(map_user_cf_per_group, label="UserCF")
        pyplot.plot(map_fallback_per_group, label="Fallback")
        pyplot.plot(map_fallback_with_hstack_per_group, label="Fallback (hstack)")
        pyplot.plot(map_hybrid_per_group, label="Hybrid")

        pyplot.xlabel('User Group')
        pyplot.ylabel('MAP')
        pyplot.xticks(np.arange(0, 20, 1))
        pyplot.grid(b=True, axis='both', color='firebrick', linestyle='--', linewidth=0.5)
        pyplot.legend(loc='lower right')
        pyplot.show()


if __name__ == '__main__':

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

    initial_target_users = target_users_list()

    user_wise_evaluator = UserWiseEvaluation()
    user_wise_evaluator.fit(urm_train, icm_all, ucm_all)
    user_wise_evaluator.evaluate_and_plot(urm_test, initial_target_users)
