seed = 42
# Dataset
make_REASON_binary = True

# DataLoader
batch_size = 32
ratio = [0.7, 0.15, 0.15]
num_workers = 0

# MI
n_neighbors = 3

# edge index
edge_top_k = 6
edge_threshold = 0.01
edge_pruning_ratio = 0.5



import os
import pandas as pd
import argparse
from data_processing.tensor_dataset import TEDSTensorDataset, TEDSDatasetForGIN
from data_processing.data_utils import train_test_split_stratified
from data_processing.process_mi_dict import search_mi_dict
from data_processing.edge import mi_edge_index_batched

cur_dir = os.path.dirname(__file__)
root = os.path.join(cur_dir, 'data')

dataset = TEDSTensorDataset(root, binary=make_REASON_binary)
col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info
train_loader, val_loader, test_loader, train_idx = train_test_split_stratified(
    dataset=dataset,
    batch_size=batch_size,
    ratio=ratio,
    seed=seed,
    num_workers=num_workers
)

# calculate MI
train_df = dataset.processed_df.iloc[train_idx].copy()
los = pd.Series(dataset.LOS[train_idx].numpy().flatten(), index=train_idx)
train_df['LOS'] = los
mi_ad_dict, mi_dis_dict, mi_avg_dict = search_mi_dict(root=root, seed=seed, train_df=train_df, n_neighbors=n_neighbors)

# get edge index
edge_index_1 = mi_edge_index_batched(batch_size=batch_size, 
                                     num_nodes=len(ad_col_index), 
                                     mi_ad_dict=mi_ad_dict, 
                                     mi_dis_dict=mi_dis_dict,
                                     top_k=edge_top_k,
                                     threshold=edge_threshold,
                                     pruning_ratio=edge_pruning_ratio,
                                     return_edge_attr=False)

edge_index_2 = mi_edge_index_batched(batch_size=batch_size,
                                     num_nodes=len(ad_col_index),
                                     mi_ad_dict=mi_ad_dict,
                                     mi_dis_dict=mi_dis_dict,
                                     top_k=edge_top_k,
                                     threshold=edge_threshold,
                                     pruning_ratio=edge_pruning_ratio,
                                     return_edge_attr=True)

