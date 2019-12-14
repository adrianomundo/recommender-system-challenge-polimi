import sys
import time

import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet

from utils.data_handler import check_matrix


class SLIMElasticNetRecommender(object):

    def __init__(self):

        self.urm_train = None
        self.l1_ratio = None
        self.positive_only = None
        self.topK = None
        self.model = None
        self.W_sparse = None

    def fit(self, urm_train, l1_ratio=0.1, positive_only=True, top_k=100, load_matrix=False):

        self.urm_train = urm_train

        if not load_matrix:
            assert 0 <= l1_ratio <= 1, \
                "SLIM_ElasticNet: l1_ratio must be between 0 and 1, provided value was {}".format(l1_ratio)

            self.l1_ratio = l1_ratio
            self.positive_only = positive_only
            self.topK = top_k

            # initialize the ElasticNet model
            self.model = ElasticNet(alpha=1e-4,
                                    l1_ratio=self.l1_ratio,
                                    positive=self.positive_only,
                                    fit_intercept=False,
                                    copy_X=False,
                                    precompute=True,
                                    selection='random',
                                    max_iter=100,
                                    tol=1e-4)

            urm_train = check_matrix(self.urm_train, 'csc', dtype=np.float32)

            n_items = urm_train.shape[1]

            # Use array as it reduces memory requirements compared to lists
            data_block = 10000000

            rows = np.zeros(data_block, dtype=np.int32)
            cols = np.zeros(data_block, dtype=np.int32)
            values = np.zeros(data_block, dtype=np.float32)

            num_cells = 0

            start_time = time.time()
            start_time_print_batch = start_time

            # fit each item's factors sequentially (not in parallel)
            for currentItem in range(n_items):

                # get the target column
                y = urm_train[:, currentItem].toarray()

                if y.sum() == 0.0:
                    continue

                # set the j-th column of X to zero
                start_pos = urm_train.indptr[currentItem]
                end_pos = urm_train.indptr[currentItem + 1]

                current_item_data_backup = urm_train.data[start_pos: end_pos].copy()
                urm_train.data[start_pos: end_pos] = 0.0

                # fit one ElasticNet model per column
                self.model.fit(urm_train, y)

                # self.model.coef_ contains the coefficient of the ElasticNet model
                # let's keep only the non-zero values

                # Select topK values
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index

                # nonzero_model_coef_index = self.model.coef_.nonzero()[0]
                # nonzero_model_coef_value = self.model.coef_[nonzero_model_coef_index]

                nonzero_model_coef_index = self.model.sparse_coef_.indices
                nonzero_model_coef_value = self.model.sparse_coef_.data

                local_top_k = min(len(nonzero_model_coef_value) - 1, self.topK)

                relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_top_k)[0:local_top_k]
                relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
                ranking = relevant_items_partition[relevant_items_partition_sorting]

                for index in range(len(ranking)):

                    if num_cells == len(rows):
                        rows = np.concatenate((rows, np.zeros(data_block, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(data_block, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(data_block, dtype=np.float32)))

                    rows[num_cells] = nonzero_model_coef_index[ranking[index]]
                    cols[num_cells] = currentItem
                    values[num_cells] = nonzero_model_coef_value[ranking[index]]

                    num_cells += 1

                # finally, replace the original values of the j-th column
                urm_train.data[start_pos:end_pos] = current_item_data_backup

                if time.time() - start_time_print_batch > 300 or currentItem == n_items - 1:
                    print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Items per second: {:.0f}".format(
                        currentItem + 1,
                        100.0 * float(currentItem + 1) / n_items,
                        (time.time() - start_time) / 60,
                        float(currentItem) / (time.time() - start_time)))
                    sys.stdout.flush()
                    sys.stderr.flush()

                    start_time_print_batch = time.time()

            # generate the sparse weight matrix
            self.W_sparse = sps.csr_matrix((values[:num_cells], (rows[:num_cells], cols[:num_cells])),
                                           shape=(n_items, n_items), dtype=np.float32)
            sps.save_npz("../tmp/SLIM_ElasticNet_similarity_matrix.npz", self.W_sparse)
        else:
            print("Loading SLIM_ElasticNet_similarity_matrix.npz file...")
            self.W_sparse = sps.load_npz("../tmp/SLIM_ElasticNet_similarity_matrix.npz")
            print("Matrix loaded!")

    def compute_score(self, user_id):

        user_profile = self.urm_train[user_id]

        return user_profile.dot(self.W_sparse).toarray().ravel()

    def recommend(self, user_id, at=10, exclude_seen=True):

        scores = self.compute_score(user_id)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = self.urm_train.indptr[user_id]
        end_pos = self.urm_train.indptr[user_id + 1]

        user_profile = self.urm_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores
