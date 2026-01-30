import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

def make_binary(df):
    df['REASONb'] = np.where(df['REASON'] == 1, 1, 0)
    df = df.drop(['REASON'], axis=1)
    return df

def get_col_dims(df: pd.DataFrame, ig_label: bool = False):
    col_dims = []

    for col in df.columns:
        vals = set(df[col].dropna().unique())

        if ig_label:
            vals.add(-9)  # baseline 항상 포함

        col_dims.append(len(vals))

    return col_dims

def get_ad_dis_col(df:pd.DataFrame, remove_los=False):
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

    if 'REASON' in cols:
        cols.remove('REASON')

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

def get_col_info(df: pd.DataFrame, remove_los: bool=True, ig_label: bool=False):
    '''
    Returns: (tuple)
        col_list, col_dims, ad_col_index, dis_col_index

        col_list: 보관용, 데이터에 등장하는 열 이름의 순서
        col_dims: col_list 순서대로 변수별 범주의 개수
        ad_col_index: admission에 해당하는 변수의 integer position
        dis_col_index: discharge에 해당하는 변수의 integer position
    '''
    col_list = list(df.columns)
    col_dims = get_col_dims(df, ig_label)
    ad_col_index, dis_col_index = get_ad_dis_index(df, remove_los)
    return col_list, col_dims, ad_col_index, dis_col_index

def organize_labels(df: pd.DataFrame, ig_labels: bool=False):
    '''
    -9가 있는 변수를 그대로 엔티티 임베딩에 넣으면 이상해짐
    왜냐하면 엔티티 임베딩 모델은 레이블들이 연속된 정수들의 범위로 있다고 가정하기 때문
    -9, 1, 2, 3 이렇게 있었다면
    -9, -8, -7, -6, -5, ~~~ 이런 것으로 가정함

    -9, 1, 2, 3를
    0, 1, 2, 3으로 바꿈 (-9 -> 3)
    
    + CBSA2020
    이것도 문제가 됨
    10000 24242 32646 75577 이런 식이라 연속된 정수들의 레이블이 아님
    10000 24242 32646 75577 -> 1, 2, 3, 4

    Args:
        ig_labels (bool): include a new neutral label for Integrated Gradients. Optional, default: False 
    '''
    for col in df.columns:
        if col in {"REASON", "REASONb"}:
            continue
        labels = list(df[col].unique())
        if ig_labels and (-9 not in labels):
            labels.append(-9)
        labels = sorted(labels)
        replace_dict = {v: i for i, v in enumerate(labels)}
        df[col] = df[col].replace(replace_dict)
    return df

def df_to_tensor(df: pd.DataFrame | pd.Series, dtype=torch.long):
    df_np = df.to_numpy()
    return torch.tensor(df_np, dtype=dtype)

def train_test_split_stratified(dataset, batch_size,
                              ratio=[0.7, 0.15, 0.15],
                              seed=42,
                              num_workers=0,
                              ):

    assert abs(sum(ratio) - 1.0) < 1e-6, "ratio must sum to 1.0"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 전체 인덱스 & 라벨 추출
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    indices = np.arange(len(dataset))

    unique_labels = np.unique(labels)

    train_idx = []
    val_idx = []
    test_idx = []

    # --- Stratified Split ---
    for ul in unique_labels:
        cls_idx = indices[labels == ul]
        np.random.shuffle(cls_idx)

        n_total = len(cls_idx)
        n_train = int(n_total * ratio[0])
        n_val = int(n_total * ratio[1])
        # 남은 건 test
        n_test = n_total - n_train - n_val

        # 분할
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train:n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val:])

    # 셔플 (선택)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    # Subset dataset 생성
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    print(f"Train Set Size: {len(train_dataset)}")
    print(f"Valid Set Size: {len(val_dataset)}")
    print(f"Test Set Size: {len(test_dataset)}")

    # DataLoader 생성
    # drop_last=True를 해야 마지막 자투리 배치를 위해 따로 배치 엣지 인덱스를 만들 필요가 없음
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader, (train_idx, val_idx, test_idx)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset


def make_stratified_kfold_loaders(
    dataset,
    batch_size: int,
    k_fold: int,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    drop_last: bool = True,
):
    """
    1) Stratified로 test set 분리(고정)
    2) 나머지(train pool)를 stratified하게 k개의 fold로 분할
    3) K-fold 학습에 바로 쓰도록 (train_loader, val_loader) 쌍을 생성
       - fold i를 validation으로 두고, 나머지 fold들을 합쳐 train으로 사용

    Returns:
      fold_pairs: List[Tuple[DataLoader, DataLoader]]  # (train_loader, val_loader)
      test_loader: DataLoader
      train_idx_list: List[np.ndarray]                 # idx of each train_loaders, used for building edge_index
      test_idx: np.ndarray
    """
    assert 0.0 < test_ratio < 1.0, "test_ratio must be between 0 and 1."
    assert k_fold >= 2, "k_fold must be >= 2."

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- labels / indices ---
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    indices = np.arange(len(dataset))
    unique_labels = np.unique(labels)

    train_pool_idx: list[int] = []
    test_idx: list[int] = []

    # --- stratified test split ---
    for ul in unique_labels:
        cls_idx = indices[labels == ul]
        np.random.shuffle(cls_idx)

        n_total = len(cls_idx)
        n_test = int(round(n_total * test_ratio))

        test_idx.extend(cls_idx[:n_test].tolist())
        train_pool_idx.extend(cls_idx[n_test:].tolist())

    np.random.shuffle(train_pool_idx)
    np.random.shuffle(test_idx)

    # --- stratified k-fold on train pool (round-robin per class) ---
    folds: list[list[int]] = [[] for _ in range(k_fold)]

    train_pool_idx_arr = np.array(train_pool_idx)
    train_pool_labels = labels[train_pool_idx_arr]

    for ul in unique_labels:
        cls_pool = train_pool_idx_arr[train_pool_labels == ul]
        np.random.shuffle(cls_pool)

        for j, idx in enumerate(cls_pool.tolist()):
            folds[j % k_fold].append(int(idx))

    for f in folds:
        np.random.shuffle(f)

    # --- build (train, val) loaders per fold ---
    fold_pairs = []
    train_idx_list = []
    for val_fold_id in range(k_fold):
        val_idx = folds[val_fold_id]
        train_idx = []
        for j in range(k_fold):
            if j != val_fold_id:
                train_idx.extend(folds[j])
        
        train_idx_list.append(train_idx)

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last,
        )

        fold_pairs.append((train_loader, val_loader))

    # --- fixed test loader ---
    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    print(f"{k_fold}-Fold Cross validation...")
    print(f"Train pool size (after test split): {len(train_pool_idx)}")
    print(f"Fold sizes: {[len(f) for f in folds]}")
    print(f"Test set size: {len(test_idx)}")

    return fold_pairs, test_loader, [np.array(i) for i in train_idx], np.array(test_idx)
