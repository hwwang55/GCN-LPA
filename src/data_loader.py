import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def sparse_to_tuple(sparse_matrix):
    if not sp.isspmatrix_coo(sparse_matrix):
        sparse_matrix = sparse_matrix.tocoo()
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).transpose()
    values = sparse_matrix.data
    shape = sparse_matrix.shape
    return indices, values, shape


def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.float64)


def split_dataset(n_samples):
    val_indices = np.random.choice(list(range(n_samples)), size=int(n_samples * 0.2), replace=False)
    left = set(range(n_samples)) - set(val_indices)
    test_indices = np.random.choice(list(left), size=int(n_samples * 0.2), replace=False)
    train_indices = list(left - set(test_indices))

    train_mask = get_mask(train_indices, n_samples)
    eval_mask = get_mask(val_indices, n_samples)
    test_mask = get_mask(test_indices, n_samples)

    return train_mask, eval_mask, test_mask


def load_data(dataset):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open('../data/{}/ind.{}.{}'.format(dataset, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/{}/ind.{}.test.index".format(dataset, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil().astype(np.float64)
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize_features(features)
    features = sparse_to_tuple(features)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    graph = nx.from_dict_of_lists(graph)
    graph.add_edges_from([(i, i) for i in range(len(graph.nodes)) if not graph.has_edge(i, i)])  # add self-loops
    adj = nx.adjacency_matrix(graph)
    adj = sparse_to_tuple(adj)

    train_mask, val_mask, test_mask = split_dataset(len(graph.nodes))

    return features, labels, adj, train_mask, val_mask, test_mask


def load_npz(dataset):
    file_map = {'coauthor-cs': 'ms_academic_cs.npz', 'coauthor-phy': 'ms_academic_phy.npz'}
    file_name = file_map[dataset]

    with np.load('../data/' + file_name) as f:
        f = dict(f)

        features = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']), shape=f['attr_shape'])
        features = features.astype(np.float64)
        features = normalize_features(features)
        features = sparse_to_tuple(features)

        labels = f['labels'].reshape(-1, 1)
        labels = OneHotEncoder(sparse=False).fit_transform(labels)

        adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']), shape=f['adj_shape'])
        adj += sp.eye(adj.shape[0])  # add self-loops
        adj = sparse_to_tuple(adj)

    train_mask, val_mask, test_mask = split_dataset(labels.shape[0])

    return features, labels, adj, train_mask, val_mask, test_mask


def load_random(n_nodes, n_train, n_val, p):
    features = sp.eye(n_nodes).tocsr()
    features = sparse_to_tuple(features)
    labels = np.ones([n_nodes, 1])
    graph = nx.generators.fast_gnp_random_graph(n_nodes, p=p)
    adj = nx.adjacency_matrix(graph)
    adj = sparse_to_tuple(adj)

    train_mask = np.array([1] * n_train + [0] * (n_nodes - n_train)).astype(np.float64)
    val_mask = np.array([0] * n_train + [1] * n_val + [0] * (n_nodes - n_train - n_val)).astype(np.float64)
    test_mask = np.array([0] * (n_train + n_val) + [1] * (n_nodes - n_train - n_val)).astype(np.float64)
    return features, labels, adj, train_mask, val_mask, test_mask
