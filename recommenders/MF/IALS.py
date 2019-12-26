import numpy as np
import time
import sys

from utils.data_handler import seconds_to_biggest_unit, check_matrix


class IALSRecommender(object):
    RECOMMENDER_NAME = "IALSRecommender"

    AVAILABLE_CONFIDENCE_SCALING = ["linear", "log"]

    def __init__(self):

        self.urm_train = None
        self.alpha = None
        self.num_factors = None
        self.epsilon = None
        self.reg = None
        self.warm_items = None
        self.warm_users = None
        self.regularization_diagonal = None
        self.USER_factors = None
        self.ITEM_factors = None
        self.C = None

    def fit(self, urm_train, warm_users, warm_items, epochs=10, num_factors=20, confidence_scaling="linear",
            alpha=1.0, epsilon=1.0, reg=1e-3, init_mean=0.0, init_std=0.1, **earlystopping_kwargs):

        self.urm_train = urm_train
        self.n_users = urm_train.shape[0]
        self.n_items = urm_train.shape[1]

        self.num_factors = num_factors
        self.alpha = alpha
        self.epsilon = epsilon
        self.reg = reg

        if confidence_scaling not in self.AVAILABLE_CONFIDENCE_SCALING:
            raise ValueError(
                "Value for 'confidence_scaling' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.AVAILABLE_CONFIDENCE_SCALING, confidence_scaling))

        self.USER_factors = self._init_factors(self.n_users, False)  # don't need values, will compute them
        self.ITEM_factors = self._init_factors(self.n_items)

        self._build_confidence_matrix(confidence_scaling)

        self.warm_users = warm_users
        self.warm_items = warm_items

        self.regularization_diagonal = np.diag(self.reg * np.ones(self.num_factors))

        self._update_best_model()

        self._train_with_early_stopping(epochs, algorithm_name=self.RECOMMENDER_NAME, **earlystopping_kwargs)

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best

    def compute_score(self, user_id):

        computed_score = np.dot(self.USER_factors[user_id], self.ITEM_factors.T)
        return np.squeeze(computed_score)

    def recommend(self, user_id, at=10, exclude_seen=True):

        scores = self.compute_score(user_id)

        # TODO understand unseen_warm_items -> see repo

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

    # UTILS FUNCTION NEEDED

    def _init_factors(self, num_factors, assign_values=True):
        if assign_values:
            return self.num_factors ** -0.5 * np.random.random_sample((num_factors, self.num_factors))
        else:
            return np.empty((num_factors, self.num_factors))

    def _build_confidence_matrix(self, confidence_scaling):
        if confidence_scaling == 'linear':
            self.C = self._linear_scaling_confidence()
        else:
            self.C = self._log_scaling_confidence()

        self.C_csc = check_matrix(self.C.copy(), format="csc", dtype=np.float32)

    def _linear_scaling_confidence(self):
        self.C = check_matrix(self.urm_train, format="csr", dtype=np.float32)
        self.C.data = 1.0 + self.alpha * self.C.data

        return self.C

    def _log_scaling_confidence(self):
        self.C = check_matrix(self.urm_train, format="csr", dtype=np.float32)
        self.C.data = 1.0 + self.alpha * np.log(1.0 + self.C.data / self.epsilon)

        return self.C

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()

    # EARLY STOPPING FUNCTIONS

    def _train_with_early_stopping(self, epochs_max, epochs_min=0,
                                   validation_every_n=None, stop_on_validation=False,
                                   validation_metric=None, lower_validations_allowed=None, evaluator_object=None,
                                   algorithm_name="Incremental_Training_Early_Stopping"):
        assert epochs_max > 0, "{}: Number of epochs_max must be > 0, passed was {}".format(algorithm_name, epochs_max)
        assert epochs_min >= 0, "{}: Number of epochs_min must be >= 0, passed was {}".format(algorithm_name,
                                                                                              epochs_min)
        assert epochs_min <= epochs_max, "{}: epochs_min must be <= epochs_max, passed are epochs_min {}, epochs_max {}".format(
            algorithm_name, epochs_min, epochs_max)

        # Train for max number of epochs with no validation nor early stopping
        # OR Train for max number of epochs with validation but NOT early stopping
        # OR Train for max number of epochs with validation AND early stopping
        assert evaluator_object is None or \
               (
                       evaluator_object is not None and not stop_on_validation and validation_every_n is not None and validation_metric is not None) or \
               (
                       evaluator_object is not None and stop_on_validation and validation_every_n is not None and validation_metric is not None and lower_validations_allowed is not None), \
            "{}: Inconsistent parameters passed, please check the supported uses".format(algorithm_name)

        start_time = time.time()

        self.best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self.epochs_best = 0

        epochs_current = 0

        while epochs_current < epochs_max and not convergence:

            self._run_epoch(epochs_current)

            # If no validation required, always keep the latest
            if evaluator_object is None:

                self.epochs_best = epochs_current

            # Determine whether a validaton step is required
            elif (epochs_current + 1) % validation_every_n == 0:

                print("{}: Validation begins...".format(algorithm_name))

                self._prepare_model_for_validation()

                # If the evaluator validation has multiple cutoffs, choose the first one
                results_run, results_run_string = evaluator_object.evaluateRecommender(self)
                results_run = results_run[list(results_run.keys())[0]]

                print("{}: {}".format(algorithm_name, results_run_string))

                # Update optimal model
                current_metric_value = results_run[validation_metric]

                if self.best_validation_metric is None or self.best_validation_metric < current_metric_value:

                    print("{}: New best model found! Updating.".format(algorithm_name))

                    self.best_validation_metric = current_metric_value

                    self._update_best_model()

                    self.epochs_best = epochs_current + 1
                    lower_validatons_count = 0

                else:
                    lower_validatons_count += 1

                if stop_on_validation and lower_validatons_count >= lower_validations_allowed and epochs_current >= epochs_min:
                    convergence = True

                    elapsed_time = time.time() - start_time
                    new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

                    print(
                        "{}: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                            algorithm_name, epochs_current + 1, validation_metric, self.epochs_best,
                            self.best_validation_metric, new_time_value, new_time_unit))

            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            print("{}: Epoch {} of {}. Elapsed time {:.2f} {}".format(
                algorithm_name, epochs_current + 1, epochs_max, new_time_value, new_time_unit))

            epochs_current += 1

            sys.stdout.flush()
            sys.stderr.flush()

        # If no validation required, keep the latest
        if evaluator_object is None:
            self._prepare_model_for_validation()
            self._update_best_model()

        # Stop when max epochs reached and not early-stopping
        if not convergence:
            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            if evaluator_object is not None:
                print(
                    "{}: Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                        algorithm_name, epochs_current, validation_metric, self.epochs_best,
                        self.best_validation_metric, new_time_value, new_time_unit))
            else:
                print("{}: Terminating at epoch {}. Elapsed time {:.2f} {}".format(
                    algorithm_name, epochs_current, new_time_value, new_time_unit))

    def _run_epoch(self, num_epoch):

        VV = self.ITEM_factors.T.dot(self.ITEM_factors)

        for user_id in self.warm_users:
            start_pos = self.C.indptr[user_id]
            end_pos = self.C.indptr[user_id + 1]

            user_profile = self.C.indices[start_pos:end_pos]
            user_confidence = self.C.data[start_pos:end_pos]

            self.USER_factors[user_id, :] = self._update_row(user_profile, user_confidence, self.ITEM_factors, VV)

        UU = self.USER_factors.T.dot(self.USER_factors)

        for item_id in self.warm_items:
            start_pos = self.C_csc.indptr[item_id]
            end_pos = self.C_csc.indptr[item_id + 1]

            item_profile = self.C_csc.indices[start_pos:end_pos]
            item_confidence = self.C_csc.data[start_pos:end_pos]

            self.ITEM_factors[item_id, :] = self._update_row(item_profile, item_confidence, self.USER_factors, UU)

    def _update_row(self, interaction_profile, interaction_confidence, Y, YtY):
        Y_interactions = Y[interaction_profile, :]

        A = Y_interactions.T.dot(((interaction_confidence - 1) * Y_interactions.T).T)

        B = YtY + A + self.regularization_diagonal

        return np.dot(np.linalg.inv(B), Y_interactions.T.dot(interaction_confidence))

    def _prepare_model_for_validation(self):
        pass
