import sys
import time

import numpy as np
import scipy.sparse as sps
from scipy.special import expit

from utils.data_handler import similarityMatrixTopK


class SLIM_BPR(object):
    """
    This class is a python porting of the BPRSLIM algorithm in MyMediaLite written in C#
    The code is identical with no optimizations
    """

    def __init__(self, lambda_i=0.0025, lambda_j=0.00025, learning_rate=0.05):

        self.urm_train = None
        self.n_users = None
        self.n_items = None
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        self.S = None
        self.W = None

        self.normalize = False
        self.sparse_weights = False

    def update_factors(self, user_id, pos_item_id, neg_item_id):

        # Calculate current predicted score
        user_seen_items = self.urm_train[user_id].indices
        prediction = 0

        for userSeenItem in user_seen_items:
            prediction += self.S[pos_item_id, userSeenItem] - self.S[neg_item_id, userSeenItem]

        x_uij = prediction
        logistic_function = expit(-x_uij)

        # Update similarities for all items except those sampled
        for userSeenItem in user_seen_items:

            # For positive item is PLUS logistic minus lambda*S
            if pos_item_id != userSeenItem:
                update = logistic_function - self.lambda_i * self.S[pos_item_id, userSeenItem]
                self.S[pos_item_id, userSeenItem] += self.learning_rate * update

            # For positive item is MINUS logistic minus lambda*S
            if neg_item_id != userSeenItem:
                update = - logistic_function - self.lambda_j * self.S[neg_item_id, userSeenItem]
                self.S[neg_item_id, userSeenItem] += self.learning_rate * update

    def fit(self, urm_train, epochs=1, load_matrix=False):
        """
        Train SLIM wit BPR. If the model was already trained, overwrites matrix S
        :param urm_train:
        :param epochs:
        :param load_matrix:
        :return: -
        """

        self.urm_train = urm_train

        if not load_matrix:
            self.n_users = urm_train.shape[0]
            self.n_items = urm_train.shape[1]

            # Initialize similarity with random values and zero-out diagonal
            self.S = np.random.random((self.n_items, self.n_items)).astype('float32')
            self.S[np.arange(self.n_items), np.arange(self.n_items)] = 0

            start_time_train = time.time()

            for currentEpoch in range(epochs):
                start_time_epoch = time.time()

                self.epoch_iteration()
                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch + 1, epochs,
                                                                         float(time.time() - start_time_epoch) / 60))

            print("Train completed in {:.2f} minutes".format(float(time.time() - start_time_train) / 60))

            # The similarity matrix is learnt row-wise
            # To be used in the product URM*S must be transposed to be column-wise
            self.W = self.S.T
            self.W = similarityMatrixTopK(self.W, k=100)
            sps.save_npz("../tmp/SLIM_BPR_matrix.npz", self.W)

            del self.S
        else:
            print("Loading SLIM_BPR_matrix.npz file...")
            self.W = sps.load_npz("../tmp/SLIM_BPR_matrix.npz")
            print("Matrix loaded!")

    def epoch_iteration(self):

        # Get number of available interactions
        num_positive_interactions = self.urm_train.nnz

        start_time = time.time()

        # Uniform user sampling without replacement
        for numSample in range(num_positive_interactions):

            user_id, pos_item_id, neg_item_id = self.sample_triple()
            self.update_factors(user_id, pos_item_id, neg_item_id)

            if numSample % 5000 == 0:
                print("Processed {} ( {:.2f}% ) in {:.4f} seconds".format(numSample,
                                                                          100.0 * float(
                                                                              numSample) / num_positive_interactions,
                                                                          time.time() - start_time))

                sys.stderr.flush()

                start_time = time.time()

    def sample_user(self):
        """
        Sample a user that has viewed at least one and not all items
        :return: user_id
        """
        while True:

            user_id = np.random.randint(0, self.n_users)
            num_seen_items = self.urm_train[user_id].nnz

            if 0 < num_seen_items < self.n_items:
                return user_id

    def sample_item_pair(self, user_id):
        """
        Returns for the given user a random seen item and a random not seen item
        :param user_id:
        :return: pos_item_id, neg_item_id
        """

        user_seen_items = self.urm_train[user_id].indices

        pos_item_id = user_seen_items[np.random.randint(0, len(user_seen_items))]

        while True:

            neg_item_id = np.random.randint(0, self.n_items)

            if neg_item_id not in user_seen_items:
                return pos_item_id, neg_item_id

    def sample_triple(self):
        """
        Randomly samples a user and then samples randomly a seen and not seen item
        :return: user_id, pos_item_id, neg_item_id
        """

        user_id = self.sample_user()
        pos_item_id, neg_item_id = self.sample_item_pair(user_id)

        return user_id, pos_item_id, neg_item_id

    def compute_score(self, user_id):

        # compute the scores using the dot product
        user_profile = self.urm_train[user_id]
        return user_profile.dot(self.W).toarray().ravel()

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
