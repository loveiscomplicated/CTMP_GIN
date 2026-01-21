# models/factory.py
import pandas
import os
from src.data_processing.mi_dict import search_mi_dict
from src.data_processing.edge import fully_connected_edge_index_batched, mi_edge_index_batched, mi_edge_index_batched_for_a3tgcn, mi_edge_index_batched_for_gin
from src.models.ctmp_gin import CTMPGIN
from src.models.gin import GIN,  GIN_m

import torch
MODEL_REGISTRY = {
    "ctmp_gin": CTMPGIN,
    "gin": GIN,
    "gin_m": GIN_m
}

def build_model(model_name: str, device: torch.device, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](device=device, **kwargs)

def build_edge(model_name: str, 
               root: str,
               seed: int,
               train_df: pandas.DataFrame,
               num_nodes: int,
               batch_size: int,
               **kwargs):
    if not kwargs["is_mi_based"]:
        return fully_connected_edge_index_batched(num_nodes=num_nodes, batch_size=batch_size)
    
    mi_ad_dict, mi_dis_dict, mi_avg_dict, mi_dict = search_mi_dict(root=root,
                                seed=seed,
                                train_df=train_df,
                                n_neighbors=kwargs['n_neighbors'])
    if model_name == "a3tgcn":
        edge_index = mi_edge_index_batched_for_a3tgcn(batch_size=batch_size, 
                                                        num_nodes=num_nodes, 
                                                        mi_avg_dict=mi_avg_dict,
                                                        top_k=kwargs["top_k"],
                                                        threshold=kwargs["threshold"],
                                                        pruning_ratio=kwargs["pruning_ratio"],
                                                        return_edge_attr=kwargs["return_edge_attr"])
        return edge_index
    
    if model_name == "gin":
        edge_index = mi_edge_index_batched_for_gin(batch_size=batch_size, 
                                                        num_nodes=num_nodes, 
                                                        mi_dict_all_variables=mi_dict,
                                                        top_k=kwargs["top_k"],
                                                        threshold=kwargs["threshold"],
                                                        pruning_ratio=kwargs["pruning_ratio"],
                                                        return_edge_attr=kwargs["return_edge_attr"])
        return edge_index
    
    edge_index = mi_edge_index_batched(batch_size=batch_size,
                                       num_nodes=num_nodes,
                                       mi_ad_dict=mi_ad_dict,
                                       mi_dis_dict=mi_dis_dict,
                                       top_k=kwargs["top_k"],
                                       threshold=kwargs["threshold"],
                                       pruning_ratio=kwargs["pruning_ratio"],
                                       return_edge_attr=kwargs["return_edge_attr"])
    return edge_index

