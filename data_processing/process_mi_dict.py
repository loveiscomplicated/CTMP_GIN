import os
import pickle
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from .data_utils import get_ad_dis_col

'''
1. search, check whether mi_dict exists
2. if exists, load (pickle)
3. else, calculate
(value of mi_dict must be pd.Series)
    df_ad, df_dis
    get mi of each.
'''

def _get_mi_helper(df: pd.DataFrame, seed: int, n_neighbors):
    mi_dict = {}
    for col in tqdm(df.columns):
        x = df.drop(col, axis=1)
        y = df[col]
        mi = mutual_info_classif(x, y, discrete_features=True, n_neighbors=n_neighbors, random_state=seed)
        mi_series = pd.Series(mi, index=x.columns)
        mi_dict[col] = mi_series
    return mi_dict

def get_mi_dict(train_df: pd.DataFrame, seed: int, mi_dict_path: str, n_neighbors=3):
    mi_dict = _get_mi_helper(train_df, seed, n_neighbors)
    with open(mi_dict_path, 'wb') as f:
        pickle.dump(mi_dict, f)
    
    return mi_dict

def _seperate_ad(mi_dict: dict, ad_col_list):
    mi_ad_dict = {}
    for key, value in mi_dict.items():
        if key not in ad_col_list:
            continue
        cur_col = deepcopy(ad_col_list)
        cur_col.remove(key)
        new_value = value[cur_col]
        mi_ad_dict[key] = new_value
    return mi_ad_dict

def _seperate_dis(mi_dict: dict, dis_col_list):
    mi_dis_dict = {}
    for key, value in mi_dict.items():
        if key in dis_col_list:
            cur_col = deepcopy(dis_col_list)
            cur_col.remove(key)
            new_value = value[cur_col]
            new_index = [i[:-2] if "_D" in i else i for i in new_value.index]
            new_value = value.reindex(new_index)

            if "_D" in key:
                key = key[:-2]
            
            mi_dis_dict[key] = new_value
    return mi_dis_dict

def _get_avg(mi_ad_dict: dict, mi_dis_dict: dict):
    mi_avg_dict = {}
    var_list = mi_ad_dict.keys()
    
    for var in var_list:
        avg_value = (mi_ad_dict[var] + mi_dis_dict[var]) / 2
        mi_avg_dict[var] = avg_value
    
    return mi_avg_dict

def seperate_ad_dis(mi_dict: dict, ad_col_list, dis_col_list):
    mi_ad_dict = _seperate_ad(mi_dict=mi_dict, ad_col_list=ad_col_list)
    mi_dis_dict = _seperate_dis(mi_dict=mi_dict, dis_col_list=dis_col_list)
    mi_avg_dict = _get_avg(mi_ad_dict=mi_ad_dict, mi_dis_dict=mi_dis_dict)
    return mi_ad_dict, mi_dis_dict, mi_avg_dict


def search_mi_dict(root: str, seed: int, train_df: pd.DataFrame, n_neighbors=3):
    mi_dict_path = os.path.join(root, 'mi', f'mi_dict_{seed}.pickle')
    
    if os.path.exists(mi_dict_path):
        print("Loading Cached file...")
        with open(mi_dict_path, 'rb') as f:
            mi_dict = pickle.load(f)
    else:
        print("Calculating MI...")
        mi_dict = get_mi_dict(train_df=train_df, seed=seed, mi_dict_path=mi_dict_path, n_neighbors=3)

    ad_col_list, dis_col_list = get_ad_dis_col(df=train_df, remove_los=True)
    mi_ad_dict, mi_dis_dict, mi_avg_dict = seperate_ad_dis(mi_dict=mi_dict, ad_col_list=ad_col_list, dis_col_list=dis_col_list)
    return mi_ad_dict, mi_dis_dict, mi_avg_dict