import anndata
import numpy as np
import scanpy as sc
import torch
import pandas as pd
import warnings
from torch_geometric.data import InMemoryDataset, Data
import json

warnings.filterwarnings("ignore", module="anndata")

class Sc(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

        with open(self.processed_paths[1], 'r') as f:
            self.size_info = json.load(f)

    @property
    def raw_file_names(self):
        return ["cancer.h5ad", "normal.h5ad"]

    @property
    def processed_file_names(self):
        return ['data.pt', 'size_info.json']

    def process(self):
        file_cancer = self.raw_paths[0]
        file_normal = self.raw_paths[1]

        adata_normal = anndata.read_h5ad(file_normal)
        adata_cancer = anndata.read_h5ad(file_cancer)

        sample_size_per_type = 50 # Number of samples taken
        samples_time = 500  # Number of sampling (number of subgraphs generated)

        normal_counts = adata_normal.obs['Celltype..major.lineage.'].value_counts().sort_index()
        cancer_counts = adata_cancer.obs['Celltype..major.lineage.'].value_counts().sort_index()

        common_cell_types = normal_counts.index.intersection(cancer_counts.index)
        gene_length = adata_normal.shape[1]
        subgraph_size = len(common_cell_types) * sample_size_per_type

        # The size_info.json file is saved to record the number of cell types, the number of samples for each cell type, the size of the subgraph, and the node labels
        size_info = {
            'normal': normal_counts.to_dict(),
            'cancer': cancer_counts.to_dict(),
            'sample_size_per_type': sample_size_per_type,
            'subgraph_node_label':{cell_type: sample_size_per_type for cell_type in common_cell_types},
            'subgraph_size': subgraph_size,  
            'gene_length': gene_length
        }

        with open(self.processed_paths[1], 'w') as f:
            json.dump(size_info, f)

        data_list = []
        for i in range(samples_time):
            # Samples were taken from the normal data set
            sampled_normal = self.one_sample_data(adata_normal, sample_counts=normal_counts.index, sample_size=sample_size_per_type, seed=i)
            x_normal = torch.tensor(sampled_normal.X.toarray(), dtype=torch.float)
            y_normal = torch.tensor([1]).float().reshape(-1, 1)  # 1 means normal
            node_label_normal = torch.tensor(sampled_normal.obs["encoded_celltypes"].to_numpy(), dtype=torch.long)
            edge_normal = self.get_graph_edge(sampled_normal)
            data_list.append(Data(x=x_normal, y=y_normal, edge_index=edge_normal, node_label=node_label_normal))

            # Samples were taken from the cancer data set
            sampled_cancer = self.one_sample_data(adata_cancer, sample_counts=cancer_counts.index, sample_size=sample_size_per_type, seed=i)
            x_cancer = torch.tensor(sampled_cancer.X.toarray(), dtype=torch.float)
            y_cancer = torch.tensor([0]).float().reshape(-1, 1)  # 0 means cancer
            node_label_cancer = torch.tensor(sampled_cancer.obs["encoded_celltypes"].to_numpy(), dtype=torch.long)
            edge_cancer = self.get_graph_edge(sampled_cancer)
            data_list.append(Data(x=x_cancer, y=y_cancer, edge_index=edge_cancer, node_label=node_label_cancer))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("-------------------Sampling complete------------------")

    def one_sample_data(self, adata, sample_counts, sample_size, seed=None):
        np.random.seed(seed)
        sampled_cells = []

        for cell_type in sample_counts:
            indices = adata.obs_names[adata.obs['Celltype..major.lineage.'] == cell_type].tolist()

            current_sampled_cells = np.random.choice(indices, sample_size, replace=True)
            sampled_cells.extend(current_sampled_cells)

        sampled_adata = adata[sampled_cells].copy()
        sampled_adata.obs['original_obs_names'] = sampled_adata.obs_names
        sampled_adata.obs_names = pd.Index([str(i) for i in range(sampled_adata.n_obs)])
        return sampled_adata

    def get_graph_edge(self, adata):
        sc.pp.neighbors(adata, use_rep='X')
        connectivities = adata.obsp['connectivities']

        row, col = connectivities.nonzero()
        edge_indices = np.vstack((row, col)) 
        edge_indices_tensor = torch.tensor(edge_indices, dtype=torch.long)

        return edge_indices_tensor
