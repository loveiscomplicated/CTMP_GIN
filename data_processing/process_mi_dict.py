import os
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from .data_utils import get_ad_dis_col

def search_mi_dict(root: str, seed: int, train_df: pd.DataFrame, n_neighbors=3):
    mi_ad_dict_path = os.path.join(root, f"mi_ad_dict_{seed}.pickle")
    mi_dis_dict_path = os.path.join(root, f"mi_dis_dict_{seed}.pickle")
    mi_avg_dict_path = os.path.join(root, f"mi_avg_dict_{seed}.pickle")
    
    if os.path.exists(mi_ad_dict_path):
        with open(mi_ad_dict_path, 'rb') as f:
            mi_ad_dict = pickle.load(f)
    else:
        mi_ad_dict = get_mi_dict(train_df=train_df,
                                 seed=seed,
                                 mode='ad',
                                 root=root,
                                 n_neighbors=n_neighbors)
        
    if os.path.exists(mi_dis_dict_path):
        with open(mi_dis_dict_path, 'rb') as f:
            mi_dis_dict = pickle.load(f)
    else:
        mi_dis_dict = get_mi_dict(train_df=train_df,
                                  seed=seed,
                                  mode='dis',
                                  root=root,
                                  n_neighbors=n_neighbors)

    if os.path.exists(mi_avg_dict_path):
        with open(mi_avg_dict_path, 'rb') as f:
            mi_avg_dict = pickle.load(f)
    else:
        mi_avg_dict = get_mi_avg_dict(mi_ad_dict=mi_ad_dict, # type: ignore
                                      mi_dis_dict=mi_dis_dict, # type: ignore
                                      root=root,
                                      seed=seed)
        
    return mi_ad_dict, mi_dis_dict, mi_avg_dict

def _get_mi_helper(df: pd.DataFrame, seed: int, n_neighbors):
    mi_dict = {}
    for col in tqdm(df.columns):
        x = df.drop(col, axis=1)
        y = df[col]
        mi = mutual_info_classif(x, y, discrete_features=True, n_neighbors=n_neighbors, random_state=seed)
        mi_dict[col] = mi
    return mi_dict

def get_mi_dict(train_df: pd.DataFrame, seed: int, mode: str, root:str, n_neighbors=3):
    mi_ad_dict_path = os.path.join(root, f"mi_ad_dict_{seed}.pickle")
    mi_dis_dict_path = os.path.join(root, f"mi_dis_dict_{seed}.pickle")

    ad, dis = get_ad_dis_col(train_df)
    df_ad = train_df[ad]
    df_dis = train_df[dis]

    if mode == 'ad':
        print("\nCalculating MI of admission data...") 
        mi_ad_dict = _get_mi_helper(df_ad, seed, n_neighbors)

        with open(mi_ad_dict_path, 'wb') as f:
            pickle.dump(mi_ad_dict, f)

        return mi_ad_dict
    
    if mode == 'dis':
        print("Calculating MI of discharge data...") 
        mi_dis_dict = _get_mi_helper(df_dis, seed, n_neighbors)

        with open(mi_dis_dict_path, 'wb') as f:
            pickle.dump(mi_dis_dict, f)

        return mi_dis_dict

def get_mi_avg_dict(mi_ad_dict: dict, mi_dis_dict: dict, root: str, seed: int):
    print("Averaging the results to get mi_avg_dict...")

    mi_avg_dict_path = os.path.join(root, f"mi_avg_dict_{seed}.pickle")

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

    with open(mi_avg_dict_path, 'wb') as f:
        pickle.dump(mi_avg_dict, f)

    return mi_avg_dict