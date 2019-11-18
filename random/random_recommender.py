import scipy.sparse as sps
import numpy as np
from data.results.csv_builder import create_csv

# TODO Find a better way to skip the header on the data_train.csv file
URM_path = "../data/data_train_no_header.csv"
URM_file = open(URM_path, 'r')


def row_split(row_string):
    split = row_string.split(",")
    split[2] = split[2].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])

    result = tuple(split)

    return result


URM_file.seek(0)
URM_tuples = []

for line in URM_file:
    URM_tuples.append(row_split(line))

# print(URM_tuples[0:10])

userList, itemList, ratingList = zip(*URM_tuples)

userList = list(userList)
itemList = list(itemList)
ratingList = list(ratingList)

# print(userList[0:10])
# print(itemList[0:10])
# print(ratingList[0:10])

# A sparse matrix in COOrdinate format
# A sparse matrix is a matrix in which most of the elements are zero
URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
# print(URM_all)

URM_all.tocsr()
# print(URM_all)


class RandomRecommender(object):

    def fit(self, urm_train):
        self.numItems = urm_train.shape[0]

    def recommend(self, user_id, at=5):
        recommended_items = np.random.choice(self.numItems, at)

        return recommended_items


train_test_split = 0.80

# Get the count of explicitly-stored values (non-zeros)
numInteractions = URM_all.nnz
# print(numInteractions)

# Array randomly filled by True or False values (len == numInteractions)
train_mask = np.random.choice([True, False], numInteractions, p=[train_test_split, 1 - train_test_split])
# print(np.sum(train_mask))
# print(train_mask)
# print(len(train_mask))

userList = np.array(userList)
itemList = np.array(itemList)
ratingList = np.array(ratingList)

# print(len(userList))
# print(len(itemList))
# print(len(ratingList))


URM_train = sps.coo_matrix((ratingList[train_mask], (userList[train_mask], itemList[train_mask])))
URM_train = URM_train.tocsr()

# print(URM_train)
# print(URM_train.shape[0])

test_mask = np.logical_not(train_mask)

URM_test = sps.coo_matrix((ratingList[test_mask], (userList[test_mask], itemList[test_mask])))
URM_test = URM_test.tocsr()

randomRecommender = RandomRecommender()
randomRecommender.fit(URM_train)

results = {}

userList_unique = list(set(userList))
# print(len(userList_unique))

for user in userList_unique:
    recommended_items = randomRecommender.recommend(user, at=10)
    results[user] = recommended_items

create_csv(results)
