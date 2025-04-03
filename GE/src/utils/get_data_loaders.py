import torch
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader

from datasets import Sc


def get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state):
    multi_label = False
    assert dataset_name in ['sc', ]

    if dataset_name == 'sc':
        dataset = Sc(root=data_dir / 'sc')
        split_idx = get_random_split_idx(dataset, splits)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]

    x_dim = test_set[0].x.shape[1]
    edge_attr_dim = 0 if test_set[0].edge_attr is None else test_set[0].edge_attr.shape[1]
    if isinstance(test_set, list):
        num_class = Batch.from_data_list(test_set).y.unique().shape[0]
    elif test_set.data.y.shape[-1] == 1 or len(test_set.data.y.shape) == 1:
        num_class = test_set.data.y.unique().shape[0]
    else:
        num_class = test_set.data.y.shape[-1]
        multi_label = True

    print('[INFO] Calculating degree...')

    batched_train_set = Batch.from_data_list(train_set)
    d = degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long)
    deg = torch.bincount(d, minlength=10)

    aux_info = {'deg': deg, 'multi_label': multi_label}  #
    return loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info


def get_random_split_idx(dataset, splits, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]
    test_idx = idx[n_train + n_valid:]
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, dataset_splits=None):
    if split_idx is not None:
        assert dataset is not None
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size,
                                  shuffle=True)  # shuffle=True，数据将在每个epoch重新洗牌。
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        test_set = dataset.copy(split_idx["test"])  # For visualization
    else:
        assert dataset_splits is not None
        train_loader = DataLoader(dataset_splits['train'], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_splits['valid'], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_splits['test'], batch_size=batch_size, shuffle=False)
        test_set = dataset_splits['test']  # For visualization
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, test_set
