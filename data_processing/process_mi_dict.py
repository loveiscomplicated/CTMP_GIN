import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif
from .data_utils import get_ad_dis_col

def _get_mi_helper(df: pd.DataFrame, seed: int, n_neighbors):
    mi_dict = {}
    for col in df.columns:
        print(col)
        x = df.drop(col, axis=1)
        y = df[col]
        mi = mutual_info_classif(x, y, discrete_features=True, n_neighbors=n_neighbors, random_state=seed)
        mi_dict[col] = mi
        print(f'{col} finished')
    return mi_dict

def get_mi_dict(train_df: pd.DataFrame, seed: int, n_neighbors=3):
    ad, dis = get_ad_dis_col(train_df)
    df_ad = train_df[ad]
    df_dis = train_df[dis]

    print("Calculating MI of admission data...") 
    mi_ad_dict = _get_mi_helper(df_ad, seed, n_neighbors)

    print("Calculating MI of discharge data...") 
    mi_dis_dict = _get_mi_helper(df_dis, seed, n_neighbors)

    return mi_ad_dict, mi_dis_dict

def get_mi_dict_static(train_df: pd.DataFrame, seed: int, n_neighbors=3):
    mi_ad_dict, mi_dis_dict = get_mi_dict(train_df, seed, n_neighbors)
    dis_keys_list = list(mi_dis_dict.keys())
    
    mi_avg_dict = {}

    for dis_key in dis_keys_list:
        if '_D' in dis_key:
            ad_key = dis_key[:-2]
        else:
            ad_key = dis_key

        ad_mi = mi_ad_dict[ad_key]
        dis_mi = mi_dis_dict[dis_key]
        
        avg_mi = (ad_mi + dis_mi) / 2

        mi_avg_dict[ad_key] = avg_mi

    return mi_avg_dict