import numpy as np
import implicit


class ALSRecommender(object):
    """
    ALS implemented with implicit following guideline of
    https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
    IDEA:
    Recomputing x_{u} and y_i can be done with Stochastic Gradient Descent, but this is a non-convex optimization
    problem.
    We can convert it into a set of quadratic problems, by keeping either x_u or y_i fixed while optimizing the other.
    In that case, we can iteratively solve x and y by alternating between them until the algorithm converges.
    This is Alternating Least Squares.
    """

    def __init__(self, n_factors=300, regularization=0.15, iterations=60):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.urm_train = None
        self.user_factors = None
        self.item_factors = None

    def fit(self, urm_train, save_matrix=False, load_matrix=False):

        self.urm_train = urm_train

        if not load_matrix:
            sparse_item_user = self.urm_train.T

            # Initialize the als model and fit it using the sparse item-user matrix
            model = implicit.als.AlternatingLeastSquares(factors=self.n_factors, regularization=self.regularization,
                                                         iterations=self.iterations)

            alpha_val = 24
            # Calculate the confidence by multiplying it by our alpha value.
            data_conf = (sparse_item_user * alpha_val).astype('double')
            # Fit the model
            model.fit(data_conf)

            # Get the user and item vectors from our trained model
            self.user_factors = model.user_factors
            self.item_factors = model.item_factors
            if save_matrix:
                np.savez_compressed("../tmp/IALS_USER_factors_matrix.npz", self.user_factors)
                np.savez_compressed("../tmp/IALS_ITEM_factors_matrix.npz", self.item_factors)
                print("Matrices saved!")
            else:
                print("Loading IALS_USER_factors_matrix.npz file...")
                user_factors_dict = np.load("../tmp/IALS_USER_factors_matrix.npz")
                self.user_factors = user_factors_dict['arr_0']
                print("Loading IALS_ITEM_factors_matrix.npz file...")
                item_factors_dict = np.load("../tmp/IALS_ITEM_factors_matrix.npz")
                self.item_factors = item_factors_dict['arr_0']
                print("Matrices loaded!")

    def compute_score(self, user_id):

        computed_score = np.dot(self.user_factors[user_id], self.item_factors.T)
        return np.squeeze(computed_score)

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
