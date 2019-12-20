import os
import time
from datetime import datetime
import scipy.sparse as sps
from scipy.sparse import hstack
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

from utils.IR_feature_weighting import okapi_BM_25


def data_csv_splitter(data_file):

    path = ""

    if data_file == "urm":
        path = "../data/data_train.csv"
    elif data_file == "icm_asset":
        path = "../data/data_ICM_asset.csv"
    elif data_file == "icm_price":
        path = "../data/data_ICM_price.csv"
    elif data_file == "icm_sub_class":
        path = "../data/data_ICM_sub_class.csv"
    elif data_file == "ucm_age":
        path = "../data/data_UCM_age.csv"
    elif data_file == "ucm_region":
        path = "../data/data_UCM_region.csv"

    file = open(path, 'r')

    def row_split(row_string):
        split = row_string.split(",")
        split[2] = split[2].replace("\n", "")

        split[0] = int(split[0])
        split[1] = int(split[1])
        split[2] = float(split[2])

        result = tuple(split)

        return result

    file.seek(0)
    tuples = []

    for line in file:
        if line == "row,col,data\n":
            continue
        tuples.append(row_split(line))

    return tuples


def target_users_list():

    target_path = "../data/data_target_users_test.csv"
    target_file = open(target_path, 'r')

    target_file.seek(0)
    target_list = []

    for line in target_file:
        if line == "user_id\n":
            continue
        line = int(line.rstrip('\n'))
        target_list.append(line)

    return target_list


def row_col_data_lists(tuples):

    row_list, col_list, data_list = zip(*tuples)

    row_list = list(row_list)
    col_list = list(col_list)
    data_list = list(data_list)

    return row_list, col_list, data_list


def urm_all_builder(urm_tuples):

    users_list, items_list, ratings_list = row_col_data_lists(urm_tuples)

    # A sparse matrix in COOrdinate format
    # A sparse matrix is a matrix in which most of the elements are zero
    urm_all = (sps.coo_matrix((ratings_list, (users_list, items_list)))).tocsr()

    return urm_all


def get_warm_users(urm_all):

    warm_users_mask = np.ediff1d(urm_all.tocsr().indptr) > 0
    warm_users = np.arange(urm_all.shape[0])[warm_users_mask]

    return warm_users  # , urm_all[warm_users, :]


def get_warm_items(urm_all):

    # ediff1d: the differences between consecutive elements of an array
    warm_items_mask = np.ediff1d(urm_all.tocsc().indptr) > 0
    # arange: returns the range of elements --> arange(3) = array([0, 1, 2])
    warm_items = np.arange(urm_all.shape[1])[warm_items_mask]

    return warm_items  # , urm_all[:, warm_items]


def icm_all_builder(urm_all, icm_asset_tuples, icm_price_tuples, icm_sub_class_tuples):

    row_asset, column_asset, data_asset = row_col_data_lists(icm_asset_tuples)
    row_price, column_price, data_price = row_col_data_lists(icm_price_tuples)
    row_sub_class, column_sub_class, data_sub_class = row_col_data_lists(icm_sub_class_tuples)

    item_popularity = (urm_all > 0).sum(axis=0)
    item_popularity = np.array(item_popularity).squeeze()
    row_item_popularity = np.arange(len(item_popularity))
    data_item_popularity = list(item_popularity)
    data_item_popularity = [float(item) for item in data_item_popularity]

    le_item_popularity = preprocessing.LabelEncoder()
    le_item_popularity.fit(data_item_popularity)
    data_item_popularity = le_item_popularity.transform(data_item_popularity)

    le_asset = preprocessing.LabelEncoder()
    le_asset.fit(data_asset)
    data_asset = le_asset.transform(data_asset)
    # print(data_asset[0:10])

    le_price = preprocessing.LabelEncoder()
    le_price.fit(data_price)
    data_price = le_price.transform(data_price)
    # print(data_price[0:10])

    n_items = urm_all.shape[1]
    n_features_icm_asset = max(data_asset) + 1
    n_features_icm_price = max(data_price) + 1
    n_features_icm_sub_class = max(column_sub_class) + 1
    n_features_item_popularity = max(data_item_popularity) + 1

    icm_asset_shape = (n_items, n_features_icm_asset)
    icm_price_shape = (n_items, n_features_icm_price)
    icm_sub_class_shape = (n_items, n_features_icm_sub_class)
    icm_item_popularity_shape = (n_items, n_features_item_popularity)

    ones_icm_asset = np.ones(len(data_asset))
    ones_icm_price = np.ones(len(data_price))
    ones_icm_item_popularity = np.ones(len(data_item_popularity))

    icm_asset = sps.coo_matrix((ones_icm_asset, (row_asset, data_asset)), shape=icm_asset_shape)
    icm_price = sps.coo_matrix((ones_icm_price, (row_price, data_price)), shape=icm_price_shape)
    icm_sub_class = sps.coo_matrix((data_sub_class, (row_sub_class, column_sub_class)), shape=icm_sub_class_shape)
    icm_item_popularity = sps.coo_matrix((ones_icm_item_popularity, (row_item_popularity, data_item_popularity)),
                                         shape=icm_item_popularity_shape)

    icm_all = hstack((icm_asset, icm_price))
    icm_all = hstack((icm_all, icm_sub_class))
    icm_all = hstack((icm_all, icm_item_popularity))
    icm_all = icm_all.tocsr()

    return icm_all


def bm_25_feature_weighting(icm_all):
    icm_bm25 = icm_all.astype(np.float32)
    icm_bm25 = okapi_BM_25(icm_bm25)
    return icm_bm25.tocsr()


def ucm_all_builder(urm_all, ucm_age_tuples, ucm_region_tuples):

    row_age, column_age, data_age = row_col_data_lists(ucm_age_tuples)
    row_region, column_region, data_region = row_col_data_lists(ucm_region_tuples)

    n_users = urm_all.shape[0]
    n_features_ucm_age = max(column_age) + 1
    n_features_ucm_region = max(column_region) + 1

    ucm_age_shape = (n_users, n_features_ucm_age)
    ucm_region_shape = (n_users, n_features_ucm_region)

    ucm_age = sps.coo_matrix((data_age, (row_age, column_age)), shape=ucm_age_shape)
    ucm_region = sps.coo_matrix((data_region, (row_region, column_region)), shape=ucm_region_shape)

    ucm_all = hstack((ucm_age, ucm_region))
    ucm_all = ucm_all.tocsr()

    return ucm_all


def train_test_holdout(urm_all, train_test_split=0.8):

    np.random.seed(123)

    print("train_test_split: " + str(train_test_split))

    # Get the count of explicitly-stored values (non-zeros)
    num_interactions = urm_all.nnz

    urm_all = urm_all.tocoo()
    shape = urm_all.shape

    # Array randomly filled by True or False values (len == num_interactions)
    train_mask = np.random.choice([True, False], num_interactions, p=[train_test_split, 1 - train_test_split])

    # shape=shape forces urm_train to have the same dimension of urm_all
    urm_train = sps.coo_matrix((urm_all.data[train_mask],
                                (urm_all.row[train_mask], urm_all.col[train_mask])), shape=shape)
    urm_train = urm_train.tocsr()

    test_mask = np.logical_not(train_mask)

    # shape=shape forces urm_test to have the same dimension of urm_all
    urm_test = sps.coo_matrix((urm_all.data[test_mask], (urm_all.row[test_mask], urm_all.col[test_mask])), shape=shape)
    urm_test = urm_test.tocsr()

    return urm_train, urm_test


def train_test_loo(urm_all):

    users = urm_all.shape[0]
    items = urm_all.shape[1]

    urm_train = urm_all.copy()
    urm_test = np.zeros((users, items))

    for user_id in tqdm(range(users)):
        num_interactions = urm_train[user_id].nnz
        if num_interactions > 0:
            user_profile = urm_train[user_id].indices
            item_id = np.random.choice(user_profile, 1)
            urm_train[user_id, item_id] = 0.0
            urm_test[user_id, item_id] = 1.0

    urm_train = (sps.coo_matrix(urm_train, dtype=int, shape=urm_all.shape)).tocsr()
    urm_test = (sps.coo_matrix(urm_test, dtype=int, shape=urm_all.shape)).tocsr()

    urm_train.eliminate_zeros()
    urm_test.eliminate_zeros()

    return urm_train, urm_test


def similarityMatrixTopK(item_weights, forceSparseOutput = True, k=100, verbose = False, inplace=True):
    """
    The function selects the TopK most similar elements, column-wise

    :param item_weights:
    :param forceSparseOutput:
    :param k:
    :param verbose:
    :param inplace: Default True, WARNING matrix will be modified
    :return:
    """

    assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

    start_time = time.time()

    if verbose:
        print("Generating topK matrix")

    nitems = item_weights.shape[1]
    k = min(k, nitems)

    # for each column, keep only the top-k scored items
    sparse_weights = not isinstance(item_weights, np.ndarray)

    if not sparse_weights:

        idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column

        if inplace:
            W = item_weights
        else:
            W = item_weights.copy()

        # index of the items that don't belong to the top-k similar items of each column
        not_top_k = idx_sorted[:-k, :]
        # use numpy fancy indexing to zero-out the values in sim without using a for loop
        W[not_top_k, np.arange(nitems)] = 0.0

        if forceSparseOutput:
            W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))

            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W_sparse

        if verbose:
            print("Dense TopK matrix generated in {:.2f} seconds".format(time.time()-start_time))

        return W

    else:
        # iterate over each column and keep only the top-k similar items
        data, rows_indices, cols_indptr = [], [], []

        item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)

        for item_idx in range(nitems):

            cols_indptr.append(len(data))

            start_position = item_weights.indptr[item_idx]
            end_position = item_weights.indptr[item_idx+1]

            column_data = item_weights.data[start_position:end_position]
            column_row_index = item_weights.indices[start_position:end_position]

            non_zero_data = column_data!=0

            idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
            top_k_idx = idx_sorted[-k:]

            data.extend(column_data[non_zero_data][top_k_idx])
            rows_indices.extend(column_row_index[non_zero_data][top_k_idx])


        cols_indptr.append(len(data))

        # During testing CSR is faster
        W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)
        W_sparse = W_sparse.tocsr()

        if verbose:
            print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

        return W_sparse


def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)


def create_csv(results, results_dir='../data/results'):

    csv_file_name = 'results_'
    csv_file_name += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_file_name), 'w') as f:

        f.write('user_id,item_list')

        for key, value in results.items():
            f.write('\n' + str(key) + ',')
            i = 0
            for val in value:
                f.write(str(val))
                if i != 9:
                    f.write(' ')
                    i += 1


def seconds_to_biggest_unit(time_in_seconds):

    conversion_factor = [
        ("sec", 60),
        ("min", 60),
        ("hour", 24),
        ("day", 365),
    ]

    terminate = False
    unit_index = 0

    new_time_value = time_in_seconds
    new_time_unit = "sec"

    while not terminate:

        next_time = new_time_value / conversion_factor[unit_index][1]

        if next_time >= 1.0:
            new_time_value = next_time

            unit_index += 1
            new_time_unit = conversion_factor[unit_index][0]

        else:
            terminate = True

    return new_time_value, new_time_unit
