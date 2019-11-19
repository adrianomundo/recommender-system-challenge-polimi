import scipy.sparse as sps
import numpy as np
from data.results.csv_builder import create_csv
from utils.data_handler import data_train_splitter

URM_tuples = data_train_splitter()

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

URM_all = URM_all.tocsr()
# print(URM_all)


class TopPopRecommender(object):

    def fit(self, URM_train):

        self.URM_train = URM_train

        # (URM_train > 0) --> applies a filter to URM_train:
        # if an element of URM_train is > 0 --> True
        # if an element of URM_train is < 0 --> False
        # URM_train becomes a matrix of boolean values
        # sum(axis=0) --> sums all the item in each column
        itemPopularity = (URM_train > 0).sum(axis=0)
        # squeeze to eliminate the extra dimension (dimension lost in the step above)
        itemPopularity = np.array(itemPopularity).squeeze()

        # argsort returns the indices that would sort an array
        # in this case the items' id
        self.popularItems = np.argsort(itemPopularity)
        # flip inverts the array (from the most popular to the less popular)
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self, user_id, at=5, remove_seen=True):

        if remove_seen:

            # in1d --> tests whether each element of a 1-D array is also present in a second array
            # invert=XXX --> if True, the values in the returned array are inverted
            # (that is, False where an element of ar1 is in ar2 and True otherwise)
            unseen_items_mask = np.in1d(self.popularItems, self.URM_train[user_id].indices,
                                        assume_unique=True, invert=True)

            # applies the mask in order to eliminate the already seen elements
            unseen_items = self.popularItems[unseen_items_mask]

            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.popularItems[0:at]

        return recommended_items

# -------------------------------------------------
# URM_all split with TopPopRecommender not needed

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

# print(np.sum(URM_train))
# print(URM_train.shape[0])
# print(URM_train.shape[1])

test_mask = np.logical_not(train_mask)

URM_test = sps.coo_matrix((ratingList[test_mask], (userList[test_mask], itemList[test_mask])))
URM_test = URM_test.tocsr()

# print(np.sum(URM_test))
# print(URM_test.shape[0])
# print(URM_test.shape[1])

# -------------------------------------------------

topPopRecommender_removeSeen = TopPopRecommender()
topPopRecommender_removeSeen.fit(URM_all)

results = {}

# TODO Find a better way to skip the header on the data_target_users_test.csv file
target_path = "../data/data_target_users_test_no_header.csv"
target_file = open(target_path, 'r')
target_file.seek(0)

for line in target_file:
    line = int(line.rstrip('\n'))
    recommended_items = topPopRecommender_removeSeen.recommend(line, 10)
    results[line] = recommended_items

create_csv(results)
