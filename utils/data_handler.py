import os
from datetime import datetime
import scipy.sparse as sps
import numpy as np


def data_train_csv_splitter():

    # TODO Find a better way to skip the header on the data_train.csv file
    urm_path = "../data/data_train_no_header.csv"
    urm_file = open(urm_path, 'r')

    def row_split(row_string):
        split = row_string.split(",")
        split[2] = split[2].replace("\n", "")

        split[0] = int(split[0])
        split[1] = int(split[1])
        split[2] = float(split[2])

        result = tuple(split)

        return result

    urm_file.seek(0)
    urm_tuples = []

    for line in urm_file:
        urm_tuples.append(row_split(line))

    return urm_tuples


def target_list():

    # TODO Find a better way to skip the header on the data_target_users_test.csv file
    target_path = "../data/data_target_users_test_no_header.csv"
    target_file = open(target_path, 'r')

    target_file.seek(0)
    target_list = []

    for line in target_file:
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


def urm_splitter(urm_all, user_list, item_list, rating_list, train_test_split):

    # Get the count of explicitly-stored values (non-zeros)
    num_interactions = urm_all.nnz
    # print(num_interactions)

    # Array randomly filled by True or False values (len == num_interactions)
    train_mask = np.random.choice([True, False], num_interactions, p=[train_test_split, 1 - train_test_split])
    # print(np.sum(train_mask))
    # print(train_mask)
    # print(len(train_mask))

    user_list = np.array(user_list)
    item_list = np.array(item_list)
    rating_list = np.array(rating_list)

    urm_train = sps.coo_matrix((rating_list[train_mask], (user_list[train_mask], item_list[train_mask])))
    urm_train = urm_train.tocsr()

    # print(np.sum(urm_train))
    # print("URM_train shape " + str(urm_train.shape))
    # print(urm_train.count_nonzero())

    test_mask = np.logical_not(train_mask)

    urm_test = sps.coo_matrix((rating_list[test_mask], (user_list[test_mask], item_list[test_mask])))
    urm_test = urm_test.tocsr()

    # print(np.sum(urm_test))
    # print("URM_test shape " + str(urm_test.shape))
    # print(urm_test.count_nonzero())

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
