import torch
import pandas as pd

def get_col_dims(df: pd.DataFrame):
    '''
    변수별 범주의 개수 파악
    '''
    col_dims = [len(df[col].unique()) for col in df.columns]
    return col_dims

def get_ad_dis_col(df:pd.DataFrame, remove_los=True):
    '''
    admission 시의 컬럼, discharge 시의 컬럼을 나누어 리턴
    Args:
        df(pd.DataFrame): 원본 데이터프레임, REASONb는 자동으로 제외됨
    Returns: 
        (admission 시의 컬럼 list, discharge 시의 컬럼 list)
    '''
    cols = list(df.columns)
    if remove_los:
        if 'LOS' in cols:
            cols.remove('LOS')

    if 'REASONb' in cols:
        cols.remove('REASONb')

    change = []
    change_D = []

    for i in cols:
        if i.endswith('_D'):
            change_D.append(i)
            change.append(i[:-2])
    
    ad = [i for i in cols if i not in change_D]
    dis = ad.copy()
    for i in range(len(ad)):
        if dis[i] in change:
            dis[i] = dis[i] + '_D'

    return ad, dis

def find_indices(lst, targets):
    return [lst.index(t) if t in lst else None for t in targets]

def get_ad_dis_index(df: pd.DataFrame, remove_los=True):
    col_list = list(df.columns)
    ad, dis = get_ad_dis_col(df, remove_los)
    ad_col_index = find_indices(col_list, ad)
    dis_col_index = find_indices(col_list, dis)
    return ad_col_index, dis_col_index

def get_col_info(df: pd.DataFrame, remove_los=True):
    '''
    Returns: (tuple)
        col_list, col_dims, ad_col_index, dis_col_index

        col_list: 보관용, 데이터에 등장하는 열 이름의 순서
        col_dims: col_list 순서대로 변수별 범주의 개수
        ad_col_index: admission에 해당하는 변수의 integer position
        dis_col_index: discharge에 해당하는 변수의 integer position
    '''
    col_list = list(df.columns)
    col_dims = get_col_dims(df)
    ad_col_index, dis_col_index = get_ad_dis_index(df, remove_los)
    return col_list, col_dims, ad_col_index, dis_col_index

def organize_labels(df: pd.DataFrame):
    '''
    -9가 있는 변수를 그대로 엔티티 임베딩에 넣으면 이상해짐
    왜냐하면 엔티티 임베딩 모델은 레이블들이 연속된 정수들의 범위로 있다고 가정하기 때문
    -9, 1, 2, 3 이렇게 있었다면
    -9, -8, -7, -6, -5, ~~~ 이런 것으로 가정함

    -9, 1, 2, 3를
    0, 1, 2, 3으로 바꿈 (-9 -> 4)
    
    + CBSA2020
    이것도 문제가 됨
    10000 24242 32646 75577 이런 식이라 연속된 정수들의 레이블이 아님
    10000 24242 32646 75577 -> 1, 2, 3, 4
    '''

    for col in df.columns:
        labels = sorted(df[col].unique())
        replace_dict = {labels[i]: i for i in range(len(labels))}
        df[col] = df[col].replace(replace_dict)

    return df

def df_to_tensor(df: pd.DataFrame | pd.Series, dtype=torch.long):
    df_np = df.to_numpy()
    return torch.tensor(df_np, dtype=dtype)