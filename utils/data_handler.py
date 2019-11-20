import os
from datetime import datetime
import scipy.sparse as sps
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


def user_item_rating_lists(urm_tuples):

    user_list, item_list, rating_list = zip(*urm_tuples)

    user_list = list(user_list)
    item_list = list(item_list)
    rating_list = list(rating_list)

    return user_list, item_list, rating_list


def urm_builder(urm_tuples):

    user_list, item_list, rating_list = user_item_rating_lists(urm_tuples)

    # A sparse matrix in COOrdinate format
    # A sparse matrix is a matrix in which most of the elements are zero
    urm_all = sps.coo_matrix((rating_list, (user_list, item_list)))
    urm_all = urm_all.tocsr()

    return urm_all


def train_test_holdout(urm_all, train_test_split=0.8):

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
