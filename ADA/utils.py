import numpy as np
import scipy.sparse as sp
import torch
import itertools
from scipy import sparse
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
from sklearn.decomposition import PCA
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
import anndata
import os
import csv


def adj_matrix(graph):
    nodes = []
    for src, v in graph.items():
        nodes.extend([[src, v_] for v_ in v])
        nodes.extend([[v_, src] for v_ in v])
    nodes = [k for k, _ in itertools.groupby(sorted(nodes))]
    nodes = np.array(nodes)
    return sparse.coo_matrix((np.ones(nodes.shape[0]), (nodes[:, 0], nodes[:, 1])),
                             (len(graph), len(graph)))


def norm_x(x):
    return np.diag(np.power(x.sum(axis=1), -1).flatten()).dot(x)


def normalize(mx):
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def norm_adj_matrix(matrix):
    matrix += sparse.eye(matrix.shape[0])
    degree = np.array(matrix.sum(axis=1))
    d_sqrt = sparse.diags(np.power(degree, -0.5).flatten())
    return d_sqrt.dot(matrix).dot(d_sqrt) 


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(path="process/NSCLC_GSE143423_NSCLC_GSE150660", dataset='NSCLC_GSE143423', k=5):
    file_path = f'{path}/{dataset}.h5ad'
    adata = anndata.read_h5ad(file_path)
    X_features = adata.X
    X_label = adata.obs['encoded_celltypes']
    features = torch.FloatTensor(X_features)
    
    ''' A sparse matrix representation of the k-nearest neighbor graph is generated '''
    knn_adj = kneighbors_graph(features, n_neighbors=k, include_self=True,
                               metric='cosine')
    adj = knn_adj
    
    '''compute PPMI'''
    A_k = AggTranProbMat(adj, 3)
    PPMI_ = ComputePPMI(A_k)
    n_PPMI_ = MyScaleSimMat(PPMI_)
    n_PPMI_mx = lil_matrix(n_PPMI_)
    X_n = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj)
    labels = torch.LongTensor(X_label)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return knn_adj, adj, features, labels, X_n


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


def MyScaleSimMat(W):
    '''L1 row norm of a matrix'''
    rowsum = np.array(np.sum(W, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    W = r_mat_inv.dot(W)
    return W


def AggTranProbMat(G, step):
    '''aggregated K-step transition probality'''
    G = MyScaleSimMat(G)
    G = csc_matrix.toarray(G)
    A_k = G
    A = G
    for k in np.arange(2, step + 1):
        A_k = np.matmul(A_k, G)
        A = A + A_k / k
    return A


def ComputePPMI(A):
    '''compute PPMI, given aggregated K-step transition probality matrix as input'''
    np.fill_diagonal(A, 0)
    A = MyScaleSimMat(A)
    (p, q) = np.shape(A)
    col = np.sum(A, axis=0)
    col[col == 0] = 1
    PPMI = np.log((float(p) * A) / col[None, :])
    IdxNan = np.isnan(PPMI)
    PPMI[IdxNan] = 0
    PPMI[PPMI < 0] = 0
    return PPMI


def has_self_loop(adj_matrix):
    coo_matrix = sp.coo_matrix(adj_matrix)
    return np.any(coo_matrix.row == coo_matrix.col)


def load_adj_label_for_rec(A):
    adj_label = torch.FloatTensor(A.toarray())

    pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()
    pos_weight = np.array(pos_weight).reshape(1, 1)
    pos_weight = torch.from_numpy(pos_weight)
    norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)
    return adj_label, pos_weight, norm


def sigmoid_normalize(value):
    return 1 / (1 + torch.exp(-value))


def save_embeddings(path, z_s_pr, z_s_sh, z_t_pr, z_t_sh):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, 'z_s_pr.npy'), z_s_pr)
    np.save(os.path.join(path, 'z_s_sh.npy'), z_s_sh)
    np.save(os.path.join(path, 'z_t_pr.npy'), z_t_pr)
    np.save(os.path.join(path, 'z_t_sh.npy'), z_t_sh)


def save_pred(root_path, data_trg, y_true, y_pred, results_path):
    mapping_path = f'{root_path}/mapping.csv'
    mapping_df = pd.read_csv(mapping_path)  
    mapping_dict = pd.Series(mapping_df.encoded_celltypes.values, index=mapping_df.Cell_type).to_dict()
    reverse_mapping_dict = {v: k for k, v in mapping_dict.items()}

    y_true_labels = [reverse_mapping_dict[label] for label in y_true]
    y_pred_labels = [reverse_mapping_dict[label] for label in y_pred]
    data = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_true_labels': y_true_labels,
        'y_pred_labels': y_pred_labels
    }
    results_df = pd.DataFrame(data)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    results_df.to_csv(f'{results_path}/{data_trg}.csv', index=False)


def print_model(epoch, acc, loss, cls_loss, loss_domain, loss_recon, loss_diff):
    print(
        f'epoch: {epoch}, acc_test_t: {acc}, loss: {loss}, loss_class: {cls_loss}, '
        f'loss_domain: {loss_domain}, loss_recon: {loss_recon}, loss_diff: {loss_diff}'
    )


def read_mapping_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = lines[1:]
    encoded_celltypes = [int(line.split(',')[1].strip()) for line in lines]  
    num_classes = len(set(encoded_celltypes))
    return num_classes


def save_results_to_csv(data_src, data_trg, acc, loss, file_path):
    results = {}
    if os.path.exists(file_path):
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader, None) 
            for row in reader:
                if len(row) >= 4:
                    key = (row[0], row[1])
                    results[key] = row[2:]

    results[(data_src, data_trg)] = [acc, loss]

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Data Source', 'Data Target', 'Accuracy', 'Loss']) 
        for key, value in results.items():
            writer.writerow([key[0], key[1]] + value)
