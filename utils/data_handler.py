import os
from datetime import datetime
import scipy.sparse as sps
from scipy.sparse import hstack
import numpy as np


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


def target_list():

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


def get_warm_items(urm_all):

    # print(urm_all)
    # print(urm_all.indptr[0])
    # print(urm_all.indptr[1])
    # print(urm_all.indices[urm_all.indptr[0]:urm_all.indptr[1]])

    # print(urm_all.tocsc())
    # print(urm_all.tocsc().indptr[0])
    # print(urm_all.tocsc().indptr[1])
    # print(urm_all.tocsc().indices[urm_all.indptr[0]:urm_all.indptr[1]])

    # ediff1d: the differences between consecutive elements of an array
    warm_items_mask = np.ediff1d(urm_all.tocsc().indptr) > 0
    # arange: returns the range of elements --> arange(3) = array([0, 1, 2])
    warm_items = np.arange(urm_all.shape[1])[warm_items_mask]

    return warm_items  # , urm_all[:, warm_items]


def get_warm_users(urm_all):

    warm_users_mask = np.ediff1d(urm_all.tocsr().indptr) > 0
    warm_users = np.arange(urm_all.shape[0])[warm_users_mask]

    return warm_users  # , urm_all[warm_users, :]


def single_icm_builder(single_icm_tuples):

    items_list, attributes_list, values_list = row_col_data_lists(single_icm_tuples)

    # A sparse matrix in COOrdinate format
    # A sparse matrix is a matrix in which most of the elements are zero
    single_icm = sps.coo_matrix((values_list, (items_list, attributes_list)))
    single_icm = single_icm.tocsr()

    return single_icm


def icm_all_builder(icm_asset_tuples, icm_price_tuples, icm_sub_class_tuples):

    # icm_sub_class_tuples already has all the items
    r1, c1, d1 = row_col_data_lists(icm_asset_tuples)
    r2, c2, d2 = row_col_data_lists(icm_price_tuples)

    r1_to_r2_elements = set(r1).difference(r2)
    r2_to_r1_elements = set(r2).difference(r1)

    print("Elements in ICM_asset: " + str(len(icm_asset_tuples)))
    print("Elements in ICM_price: " + str(len(icm_price_tuples)))
    print("Elements in ICM_sub_class: " + str(len(icm_sub_class_tuples)))
    print("Missing values in r2 list:", r1_to_r2_elements)
    print("Additional values in r2 list:", r2_to_r1_elements)

    icm_price_tuples = icm_add_missing_elements(r1_to_r2_elements, icm_price_tuples)
    icm_asset_tuples = icm_add_missing_elements(r2_to_r1_elements, icm_asset_tuples)

    icm_asset_tuples = sorted(icm_asset_tuples)
    icm_price_tuples = sorted(icm_price_tuples)
    icm_sub_class_tuples = sorted(icm_sub_class_tuples)

    r1, c1, d1 = row_col_data_lists(icm_asset_tuples)
    r2, c2, d2 = row_col_data_lists(icm_price_tuples)
    r3, c3, d3 = row_col_data_lists(icm_sub_class_tuples)

    icm_asset = sps.coo_matrix((d1, (r1, c1)))
    icm_asset = icm_asset.tocsr()
    icm_price = sps.coo_matrix((d2, (r2, c2)))
    icm_price = icm_price.tocsr()
    icm_sub_class = sps.coo_matrix((d3, (r3, c3)))
    icm_sub_class = icm_sub_class.tocsr()

    icm_all = hstack((icm_asset, icm_price))
    icm_all = hstack((icm_all, icm_sub_class))

    return icm_all


def icm_add_missing_elements(missing_elements, dst):

    for element in missing_elements:
        dst.append(tuple((element, 0, 0)))

    return dst


def train_test_holdout(urm_all, train_test_split=0.8):

    print("Splitting dataset using holdout function")
    print("train_test_split: " + str(train_test_split) + "\n")

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

    print("Splitting dataset using LeaveOneOut\n")

    users = urm_all.shape[0]
    items = urm_all.shape[1]

    urm_train = urm_all.copy()
    urm_test = np.zeros((users, items))

    for user_id in range(users):
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

    # print('urm_all properties')
    # print('shape =', urm_all.shape)
    # print('nnz =', urm_all.nnz)
    # print('urm_train properties')
    # print('shape =', urm_train.shape)
    # print('nnz =', urm_train.nnz)
    # print('urm_test properties')
    # print('shape =', urm_test.shape)
    # print('nnz =', urm_test.nnz)

    return urm_train, urm_test


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
