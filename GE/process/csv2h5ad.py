import pandas as pd
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import anndata
from scipy.io import mmread
import os
import sys

# GE
if len(sys.argv) != 5:
    print("Usage: seurat2h5ad.py <base_path> <train_name> <test_name> <auxiliary_name>")
    sys.exit(1)

base_path = sys.argv[1]
train_name = sys.argv[2]
test_name = sys.argv[3]
auxiliary_name = sys.argv[4]

print(base_path)
print(train_name)
print(test_name)
print(auxiliary_name)

root_path = base_path + 'GE_process/' + train_name + '_' + test_name + '_' + auxiliary_name + '/'
sparse_train_data_path = root_path + train_name + "_sparse_data.mtx"
sparse_test_data_path = root_path + test_name + "_sparse_data.mtx"
meta_train_data_path = root_path + train_name + "_meta_data.csv"
meta_test_data_path = root_path + test_name + "_meta_data.csv"
train_gene_path = root_path + train_name + "_gene_names.csv"
test_gene_path = root_path + test_name + "_gene_names.csv"

# Read sparse matrices
train_matrix = mmread(sparse_train_data_path).tocsr()
test_matrix = mmread(sparse_test_data_path).tocsr()

# Read metadata
meta_train_data = pd.read_csv(meta_train_data_path)
meta_test_data = pd.read_csv(meta_test_data_path)

# Read gene names
train_gene_names = pd.read_csv(train_gene_path)
test_gene_names = pd.read_csv(test_gene_path)

# Create AnnData objects for training and testing datasets
train_adata = anndata.AnnData(X=train_matrix.T)
test_adata = anndata.AnnData(X=test_matrix.T)

# Set metadata for training and testing datasets
train_adata.obs = meta_train_data
test_adata.obs = meta_test_data

# Set gene names as variable features for training and testing datasets
train_adata.var = train_gene_names
test_adata.var = test_gene_names

# Add batch info
train_adata.obs['batch'] = 'train'
test_adata.obs['batch'] = 'test'

# Remove cells with 'Malignant' cell type from test dataset only
test_adata = test_adata.copy()
test_adata = test_adata[test_adata.obs['Celltype..major.lineage.'] != 'Malignant']

# Save the individual datasets
normal_save_path = root_path + 'normal.h5ad'
cancer_save_path = root_path + 'cancer.h5ad'

train_adata.write(normal_save_path)
test_adata.write(cancer_save_path)

print(f"Normal data (train) saved to: {normal_save_path}")
print(f"Cancer data (test) saved to: {cancer_save_path}")
