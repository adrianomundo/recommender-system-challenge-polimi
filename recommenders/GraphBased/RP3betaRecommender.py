import sys
import time

import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize

from utils.data_handler import similarityMatrixTopK, check_matrix


class RP3betaRecommender(object):

    def __init__(self):

        self.urm_train = None
        self.alpha = None
        self.beta = None
        self.top_k = None
        self.min_rating = None
        self.implicit = None
        self.normalize_similarity = None
        self.W_sparse = None

    def fit(self, urm_train, alpha=0.41417, beta=0.04995, top_k=54, min_rating=0, implicit=True,
            normalize_similarity=True, load_matrix=False):

        self.urm_train = urm_train

        if not load_matrix:
            self.alpha = alpha
            self.beta = beta
            self.min_rating = min_rating
            self.top_k = top_k
            self.implicit = implicit
            self.normalize_similarity = normalize_similarity

            if self.min_rating > 0:
                self.urm_train.data[self.urm_train.data < self.min_rating] = 0
                self.urm_train.eliminate_zeros()
                if self.implicit:
                    self.urm_train.data = np.ones(self.urm_train.data.size, dtype=np.float32)

            # p_ui is the row-normalized urm
            p_ui = normalize(self.urm_train, norm='l1', axis=1)

            # p_iu is the column-normalized, "boolean" urm transposed
            x_bool = self.urm_train.transpose(copy=True)
            x_bool.data = np.ones(x_bool.data.size, np.float32)

            # Taking the degree of each item to penalize top popular
            # Some rows might be zero, make sure their degree remains zero
            x_bool_sum = np.array(x_bool.sum(axis=1)).ravel()

            degree = np.zeros(self.urm_train.shape[1])

            non_zero_mask = x_bool_sum != 0.0

            degree[non_zero_mask] = np.power(x_bool_sum[non_zero_mask], -self.beta)

            # ATTENTION: axis is still 1 because i transposed before the normalization
            p_iu = normalize(x_bool, norm='l1', axis=1)
            del x_bool

            # alpha power
            if self.alpha != 1.:
                p_ui = p_ui.power(self.alpha)
                p_iu = p_iu.power(self.alpha)

            # Final matrix is computed as p_ui * p_iu * p_ui
            # Multiplication unpacked for memory usage reasons
            block_dim = 200
            d_t = p_iu

            # Use array as it reduces memory requirements compared to lists
            data_block = 10000000

            rows = np.zeros(data_block, dtype=np.int32)
            cols = np.zeros(data_block, dtype=np.int32)
            values = np.zeros(data_block, dtype=np.float32)

            num_cells = 0

            start_time = time.time()
            start_time_print_batch = start_time

            for current_block_start_row in range(0, p_ui.shape[1], block_dim):

                if current_block_start_row + block_dim > p_ui.shape[1]:
                    block_dim = p_ui.shape[1] - current_block_start_row

                similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * p_ui
                similarity_block = similarity_block.toarray()

                for row_in_block in range(block_dim):
                    row_data = np.multiply(similarity_block[row_in_block, :], degree)
                    row_data[current_block_start_row + row_in_block] = 0

                    best = row_data.argsort()[::-1][:self.top_k]

                    not_zeros_mask = row_data[best] != 0.0

                    values_to_add = row_data[best][not_zeros_mask]
                    cols_to_add = best[not_zeros_mask]

                    for index in range(len(values_to_add)):

                        if num_cells == len(rows):
                            rows = np.concatenate((rows, np.zeros(data_block, dtype=np.int32)))
                            cols = np.concatenate((cols, np.zeros(data_block, dtype=np.int32)))
                            values = np.concatenate((values, np.zeros(data_block, dtype=np.float32)))

                        rows[num_cells] = current_block_start_row + row_in_block
                        cols[num_cells] = cols_to_add[index]
                        values[num_cells] = values_to_add[index]

                        num_cells += 1

                if time.time() - start_time_print_batch > 1:
                    print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                        current_block_start_row,
                        100.0 * float(current_block_start_row) / p_ui.shape[1],
                        (time.time() - start_time) / 60,
                        float(current_block_start_row) / (time.time() - start_time)))

                    sys.stdout.flush()
                    sys.stderr.flush()

                    start_time_print_batch = time.time()

            self.W_sparse = sps.csr_matrix((values[:num_cells], (rows[:num_cells], cols[:num_cells])),
                                           shape=(p_ui.shape[1], p_ui.shape[1]))

            if self.normalize_similarity:
                self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)

            if self.top_k:
                self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.top_k)

            self.W_sparse = check_matrix(self.W_sparse, format='csr')
            sps.save_npz("../tmp/RP3beta_similarity_matrix.npz", self.W_sparse)

        else:
            print("Loading RP3beta_similarity_matrix.npz file...")
            self.W_sparse = sps.load_npz("../tmp/RP3beta_similarity_matrix.npz")
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
