import sys

import numpy as np
import scipy.sparse as sps

from utils.CythonCompiler.run_compile_subprocess import run_compile_subprocess
from utils.data_handler import similarityMatrixTopK, check_matrix


class SLIM_BPR_Cython(object):

    RECOMMENDER_NAME = "SLIM_BPR_Recommender"

    def __init__(self, verbose=True, recompile_cython=False):

        self.verbose = verbose

        self.urm_train = None
        self.n_users = None
        self.n_items = None

        self.epochs = None
        self.positive_threshold = None
        self.train_with_sparse_weights = None
        self.symmetric = None
        self.batch_size = None
        self.lambda_i = None
        self.lambda_j = None
        self.learning_rate = None
        self.top_k = None
        self.sgd_mode = None
        self.gamma = None
        self.beta_1 = None
        self.beta_2 = None

        self.cython_epoch = None
        self.S_incremental = None
        self.S_best = None

        self.W_sparse = None

        if recompile_cython:
            print("Compiling in Cython")
            self.run_compilation_script()
            print("Compilation Complete")

    def fit(self,
            urm_train,
            epochs=200,
            positive_threshold=1,
            train_with_sparse_weights=False,
            symmetric=True,
            batch_size=1, lambda_i=0.0, lambda_j=0.0, learning_rate=0.01, top_k=10,
            sgd_mode='adagrad', gamma=0.995, beta_1=0.9, beta_2=0.999,
            save_matrix=False, load_matrix=False):

        self.urm_train = urm_train

        if not load_matrix:
            self.n_users = self.urm_train.shape[0]
            self.n_items = self.urm_train.shape[1]

            self.epochs = epochs
            self.positive_threshold = positive_threshold
            self.train_with_sparse_weights = train_with_sparse_weights
            self.symmetric = symmetric
            self.batch_size = batch_size
            self.lambda_i = lambda_i
            self.lambda_j = lambda_j
            self.learning_rate = learning_rate
            self.top_k = top_k
            self.sgd_mode = sgd_mode
            self.gamma = gamma
            self.beta_1 = beta_1
            self.beta_2 = beta_2

            # Select only positive interactions
            urm_train_positive = self.urm_train.copy()

            if self.positive_threshold is not None:
                urm_train_positive.data = urm_train_positive.data >= self.positive_threshold
                urm_train_positive.eliminate_zeros()

                assert urm_train_positive.nnz > 0, "SLIM_BPR_Cython: urm_train_positive is empty," \
                                                   " positive threshold is too high"

            if not self.train_with_sparse_weights:

                n_items = self.urm_train.shape[1]
                required_gb = 8 * n_items ** 2 / 1e+06

                if self.symmetric:
                    required_gb /= 2

                print("SLIM_BPR_Cython: Estimated memory required for similarity matrix of {} items is "
                      "{:.2f} MB".format(n_items, required_gb))

            # Import compiled module
            from recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython_Epoch import SLIM_BPR_Cython_Epoch

            self.cython_epoch = SLIM_BPR_Cython_Epoch(urm_train_positive,
                                                      train_with_sparse_weights=self.train_with_sparse_weights,
                                                      final_model_sparse_weights=True,
                                                      topK=self.top_k,
                                                      learning_rate=self.learning_rate,
                                                      li_reg=self.lambda_i,
                                                      lj_reg=self.lambda_j,
                                                      batch_size=self.batch_size,
                                                      symmetric=self.symmetric,
                                                      sgd_mode=self.sgd_mode,
                                                      verbose=self.verbose,
                                                      gamma=self.gamma,
                                                      beta_1=self.beta_1,
                                                      beta_2=self.beta_2)

            self._initialize_incremental_model()
            current_epoch = 0

            while current_epoch < self.epochs:
                self._run_epoch()
                self._update_best_model()
                current_epoch += 1

            self.get_S_incremental_and_set_W()

            if save_matrix:
                sps.save_npz("../tmp/SLIM_BPR_Cython_similarity_matrix.npz", self.W_sparse)
                print("Matrix saved!")

            self.cython_epoch._dealloc()

            sys.stdout.flush()
        else:
            print("Loading SLIM_BPR_Cython_similarity_matrix.npz file...")
            self.W_sparse = sps.load_npz("../tmp/SLIM_BPR_Cython_similarity_matrix.npz")
            print("Matrix loaded!")

    def _initialize_incremental_model(self):
        self.S_incremental = self.cython_epoch.get_S()
        self.S_best = self.S_incremental.copy()

    def _update_best_model(self):
        self.S_best = self.S_incremental.copy()

    def _run_epoch(self):
        self.cython_epoch.epochIteration_Cython()

    def get_S_incremental_and_set_W(self):

        self.S_incremental = self.cython_epoch.get_S()

        if self.train_with_sparse_weights:
            self.W_sparse = self.S_incremental
            self.W_sparse = check_matrix(self.W_sparse, format='csr')
        else:
            self.W_sparse = similarityMatrixTopK(self.S_incremental, k=self.top_k)
            self.W_sparse = check_matrix(self.W_sparse, format='csr')

    def compute_score(self, user_id):

        # compute the scores using the dot product
        user_profile = self.urm_train[user_id]
        return user_profile.dot(self.W_sparse).toarray().ravel()

    def recommend(self, user_id, at=10, exclude_seen=True):

        scores = self.compute_score(user_id)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = self.urm_train.indptr[user_id]
        end_pos = self.urm_train.indptr[user_id + 1]

        user_profile = self.urm_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def run_compilation_script(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        file_subfolder = "../recommenders/SLIM_BPR/Cython"
        file_to_compile_list = ['SLIM_BPR_Cython_Epoch.pyx']

        run_compile_subprocess(file_subfolder, file_to_compile_list)

        print("{}: Compiled module {} in subfolder: {}".format(self.RECOMMENDER_NAME, file_to_compile_list,
                                                               file_subfolder))

        # Command to run compilation script
        # python compile_script.py SLIM_BPR_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        # cython -a SLIM_BPR_Cython_Epoch.pyx
