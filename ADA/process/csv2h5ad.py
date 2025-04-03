import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata
import scanpy as sc
import sys
import re

# ADA
# Get command line arguments
if len(sys.argv) != 4:
    print("Usage: cs2h5ad.py <base_path> <train_name> <test_name>")
    sys.exit(1)

base_path = sys.argv[1]
train_name = sys.argv[2]
test_name = sys.argv[3]

print(base_path)
print(train_name)
print(test_name)

train_num = int(re.sub(r".*([A-Za-z]+)([0-9]+).*", r"\2", train_name))
test_num = int(re.sub(r".*([A-Za-z]+)([0-9]+).*", r"\2", test_name))

# Compare the numbers and create ordered names
if train_num < test_num:
    ordered_names = f"{train_name}_{test_name}"
else:
    ordered_names = f"{test_name}_{train_name}"

root_path = f"{base_path}ADA_process/{ordered_names}/"
norm_train_data_path = root_path + train_name + "_norm_data.csv"
norm_test_data_path = root_path + test_name + "_norm_data.csv"
meta_train_data_path = root_path + train_name + "_meta_data.csv"
meta_test_data_path = root_path + test_name + "_meta_data.csv"
gene_path = root_path + "variable_genes.csv"

# Read gene expression matrices
norm_train_data = pd.read_csv(norm_train_data_path, index_col=0)
norm_test_data = pd.read_csv(norm_test_data_path, index_col=0)

# Transpose matrices
train_matrix = norm_train_data.transpose()
test_matrix  = norm_test_data.transpose()

# Read metadata
meta_train_data = pd.read_csv(meta_train_data_path)
meta_test_data = pd.read_csv(meta_test_data_path)
gene_type = pd.read_csv(gene_path)

# Create AnnData objects
train_adata = anndata.AnnData(X=train_matrix)
test_adata = anndata.AnnData(X=test_matrix)

# Set metadata
train_adata.obs = meta_train_data.set_index('Cell')
test_adata.obs = meta_test_data.set_index('Cell')
train_adata.var = gene_type
test_adata.var = gene_type

# Add batch info
train_adata.obs['batch'] = 'train'
test_adata.obs['batch'] = 'test'

# Combine datasets
adata_combined = anndata.concat([train_adata, test_adata], join='outer', merge='unique')
sc.pp.combat(adata_combined, key='batch')

# Split and save datasets
train = adata_combined[adata_combined.obs['batch'] == 'train'].copy()
test = adata_combined[adata_combined.obs['batch'] == 'test'].copy()
train_save_path = root_path + train_name + '.h5ad'
test_save_path = root_path + test_name + '.h5ad'
train.write(train_save_path)
test.write(test_save_path)
print(f"Train data saved to: {train_save_path}")
print(f"Test data saved to: {test_save_path}")
