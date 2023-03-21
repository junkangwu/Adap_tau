import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
import warnings
import os
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)


def read_cf_amazon(file_name):
    return np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]

def read_cf_yelp2018(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    # print(len(inter_mat))
    return np.array(inter_mat)

def swap_train_sp_mat(valid_user_set, test_user_set):
    # valid
    valid_uid2swap_idx = np.array([None] * n_users)
    valid_uid2rev_swap_idx = np.array([None] * n_users)
    # valid_pos_len_list = np.zeros(n_users)
    # test_pos_len_list = np.zeros(n_users)
    valid_pos_len_list = []
    for uid in valid_user_set:
        positive_item = valid_user_set[uid]
        postive_item_num = len(positive_item)
        swap_idx = torch.FloatTensor(sorted(set(range(postive_item_num)) ^ set(positive_item)))
        valid_uid2swap_idx[uid] = swap_idx
        valid_uid2rev_swap_idx[uid] = swap_idx.flip(0)
        # valid_pos_len_list[uid] = postive_item_num
        valid_pos_len_list.append(postive_item_num)
    # test
    test_uid2swap_idx = np.array([None] * n_users)
    test_uid2rev_swap_idx = np.array([None] * n_users)
    test_pos_len_list = []
    for uid in test_user_set:
        positive_item = test_user_set[uid]
        postive_item_num = len(positive_item)
        swap_idx = torch.FloatTensor(sorted(set(range(postive_item_num)) ^ set(positive_item)))
        test_uid2swap_idx[uid] = swap_idx
        test_uid2rev_swap_idx[uid] = swap_idx.flip(0)
        # test_pos_len_list[uid] = postive_item_num
        test_pos_len_list.append(postive_item_num)

    return (valid_uid2swap_idx, valid_uid2rev_swap_idx, valid_pos_len_list), \
            (test_uid2swap_idx, test_uid2rev_swap_idx, test_pos_len_list)


def statistics(train_data, valid_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1

    if dataset == 'ali' or dataset == 'amazon':
        n_items -= n_users
        # remap [n_users, n_users+n_items] to [0, n_items]
        train_data[:, 1] -= n_users
        valid_data[:, 1] -= n_users
        test_data[:, 1] -= n_users

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))

    train_sp_mat = sp.csr_matrix((np.ones_like(train_data[:, 0]),
                                    (train_data[:, 0], train_data[:, 1])), dtype='float64', shape=(n_users, n_items))
    valid_sp_mat = sp.csr_matrix((np.ones_like(valid_data[:, 0]),
                                    (valid_data[:, 0], valid_data[:, 1])), dtype='float64', shape=(n_users, n_items))
    test_sp_mat =sp.csr_matrix((np.ones_like(test_data[:, 0]), 
                                    (test_data[:, 0], test_data[:, 1])), dtype='float64', shape=(n_users, n_items))
    # prepare for top k accerate
    valid_pre, test_pre = swap_train_sp_mat(valid_user_set, test_user_set)
    # print(len(train_user_set))
    # print(train_sp_mat.sum())
#     print()
    return train_sp_mat, valid_sp_mat, test_sp_mat, valid_pre, test_pre

def build_sparse_graph(data_cf):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_users+n_items)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items))
    return _bi_norm_lap(mat)


def load_data(model_args, logger):
    global args, dataset
    args = model_args
    dataset = args.dataset
    directory = args.data_path + dataset + '/'

    if dataset == 'yelp2018' or dataset == "amazon-book" or dataset == "gowalla" or dataset == 'ml' or dataset == "citeulike-new" or dataset == "ali-new":
        read_cf = read_cf_yelp2018
    else:
        read_cf = read_cf_amazon

    # read_cf = read_cf_yelp2018
    train_cf = read_cf(directory + 'train.txt')
    logger.info("load train.txt")
    test_cf = read_cf(directory + 'test.txt')
    if dataset == 'yelp2018' or dataset == "amazon-book" or dataset == "gowalla" or dataset == 'ml' or dataset == "citeulike-new":
        valid_cf = test_cf
    else:
        valid_cf = read_cf(directory + 'valid.txt')
    train_sp_mat, valid_sp_mat, test_sp_mat, valid_pre, test_pre = statistics(train_cf, valid_cf, test_cf)

    logger.info('building the adj mat ...')
    norm_mat = build_sparse_graph(train_cf)
    logger.info("train set len is {}".format(len(train_cf)))
    logger.info("test set len is {}".format(len(valid_cf)))

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
    }
    user_dict = {
        'train_user_set': train_user_set,
        'valid_user_set': valid_user_set if args.dataset != 'yelp2018' else None,
        'test_user_set': test_user_set,
    }
    sp_matrix = {
        'train_sp_mat': train_sp_mat,
        'valid_sp_mat': valid_sp_mat,
        'test_sp_mat': test_sp_mat
    }
    logger.info('loading over ...')
    logger.info("users are {} items are {}".format(n_users, n_items))

    return train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, None

