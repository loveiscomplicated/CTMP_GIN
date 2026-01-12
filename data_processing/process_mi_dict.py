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
    print("\nCalculating MI...") 

    mi_dict = _get_mi_helper(train_df, seed, n_neighbors)
    with open(mi_dict_path, 'wb') as f:
        pickle.dump(mi_dict, f)
    
    return mi_dict

def remove_d(mi_dis_dict: dict):
    new_dict = {}
    for key, value in mi_dis_dict.items():
        # remove "_D" suffix in key
        if "_D" in key:
            new_key = key[:-2]
        
        # remove "_D" suffix in key
        # type(value) is pandas.Series
        new_index = [i[:-2] if "_D" in i else i for i in value.index]
        new_value = value.reindex(new_index)

        new_dict[new_key] = new_value
    return new_dict

def get_mi_avg_dict(mi_ad_dict: dict, mi_dis_dict: dict):
    print("Averaging the results to get mi_avg_dict...")

def search_mi_dict(root: str, seed: int, train_df: pd.DataFrame, n_neighbors=3):
    '''
    root: path of 'data' folder
    '''
    mi_ad_dict_path = os.path.join(root, 'mi', f"mi_ad_dict_{seed}.pickle")
    mi_dis_dict_path = os.path.join(root, 'mi', f"mi_dis_dict_{seed}.pickle")
    mi_avg_dict_path = os.path.join(root, 'mi', f"mi_avg_dict_{seed}.pickle")
    
    ad_col_list, dis_col_list = get_ad_dis_col(train_df)
    
    ad_train_df = train_df[ad_col_list]
    dis_train_df = train_df[dis_col_list]

    # admission mi_dict
    if os.path.exists(mi_ad_dict_path):
        with open(mi_ad_dict_path, 'rb') as f:
            mi_ad_dict = pickle.load(f)
    else:
        
        mi_ad_dict = get_mi_dict(train_df=ad_train_df,
                                 seed=seed,
                                 mi_dict_path=mi_ad_dict_path,
                                 n_neighbors=n_neighbors)
    
    # discharge mi_dict
    if os.path.exists(mi_dis_dict_path):
        with open(mi_dis_dict_path, 'rb') as f:
            mi_dis_dict = pickle.load(f)
    else:
        mi_dis_dict = get_mi_dict(train_df=dis_train_df,
                                  seed=seed,
                                  mi_dict_path=mi_dis_dict_path,
                                  n_neighbors=n_neighbors)
        mi_dis_dict = remove_d(mi_dis_dict)

    # average mi_dict
    if os.path.exists(mi_avg_dict_path):
        with open(mi_avg_dict_path, 'rb') as f:
            mi_avg_dict = pickle.load(f)
    else:
        mi_avg_dict = get_mi_avg_dict(mi_ad_dict=mi_ad_dict, # type: ignore
                                      mi_dis_dict=mi_dis_dict, # type: ignore
                                      )
        
    return mi_ad_dict, mi_dis_dict, mi_avg_dict






def get_mi_dict_main(root: str, seed: int, train_df: pd.DataFrame, n_neighbors=3):
    mi_dict_path = os.path.join(root, 'mi', f'mi_dict_{seed}.pickle')
    
    if os.path.exists(mi_dict_path):
        with open(mi_dict_path, 'rb') as f:
            mi_dict = pickle.load(f)
    else:
        mi_dict = get_mi_dict(train_df=train_df, seed=seed, mi_dict_path=mi_dict_path, n_neighbors=3)

        