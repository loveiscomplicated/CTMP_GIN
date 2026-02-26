import os
import sys
import time
import json
import pickle
import subprocess
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.data_utils import train_test_split_stratified
from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.utils.device_set import device_set

REMOTE_BASE = "gdrive:CTMP_GIN_mi_service"
LOCAL_CACHE = os.path.expanduser("~/mi_cache")
REQUESTS_DIR = f"{REMOTE_BASE}/requests"
RESPONSES_DIR = f"{REMOTE_BASE}/responses"
DONE_DIR = f"{REMOTE_BASE}/done"

os.makedirs(LOCAL_CACHE, exist_ok=True)

def rclone_copy(src, dst):
    subprocess.run(["rclone", "copyto", src, dst], check=True)


def rclone_move(src, dst):
    subprocess.run(["rclone", "moveto", src, dst], check=True)


def rclone_list(remote_path):
    result = subprocess.run(
        ["rclone", "lsf", remote_path],
        capture_output=True,
        text=True
    )
    return result.stdout.splitlines()


def _get_mi_helper(df: pd.DataFrame, seed: int, n_neighbors):
    mi_dict = {}
    for col in tqdm(df.columns):
        x = df.drop(col, axis=1)
        y = df[col]
        mi = mutual_info_classif(
            x,
            y,
            discrete_features=True,
            n_neighbors=n_neighbors,
            random_state=seed
        )
        mi_series = pd.Series(mi, index=x.columns)
        mi_dict[col] = mi_series
    return mi_dict


# -------------------------
# 실제 train 데이터 로드 함수
# -------------------------
def load_train_df(mode, fold, seed, cfg):
    cur_dir = os.path.dirname(__file__)
    root = os.path.join(cur_dir, '..', 'src', 'data')

    if mode == "single":
        dataset = TEDSTensorDataset(
                root=root,
                binary=cfg["train"].get("binary", True),
                ig_label=cfg["train"].get("ig_label", False),
            )

        cfg["model"]["params"]["col_info"] = dataset.col_info
        cfg["model"]["params"]["num_classes"] = dataset.num_classes
        device = device_set(cfg["device"])

        cfg["model"]["params"]["device"] = device
        
        num_nodes = len(dataset.col_info[2]) # col_info: (col_list, col_dims, ad_col_index, dis_col_index)

        if cfg["model"]["name"] == 'gin':
            num_nodes = len(dataset.col_info[0]) + 1

        print(f"num_nodes set to {num_nodes}")

        # create dataloaders
        split_ratio = [cfg['train']['train_ratio'], cfg['train']['val_ratio'], cfg['train']['test_ratio']]
        train_loader, val_loader, test_loader, idx = train_test_split_stratified(dataset=dataset,  # type: ignore
                                                                                    batch_size=cfg['train']['batch_size'],
                                                                                    ratio=split_ratio,
                                                                                    seed=seed,
                                                                                    num_workers=cfg['train']['num_workers'],
                                                                                    )
        train_df = dataset.processed_df.iloc[idx[0]]
        return train_df

    elif mode == "cv":
        raise KeyError("not implemented yet")

# -------------------------
# 메인 루프
# -------------------------
def main():
    print("MI worker started...")

    while True:
        try:
            request_files = rclone_list(REQUESTS_DIR)

            for fname in request_files:
                if not fname.endswith(".json"):
                    continue

                remote_request = f"{REQUESTS_DIR}/{fname}"
                local_request = os.path.join("/tmp", fname)

                print(f"Processing {fname}")
                rclone_copy(remote_request, local_request)

                with open(local_request) as f:
                    req = json.load(f)

                artifact_key = req["artifact_key"]
                local_cache_path = os.path.join(
                    LOCAL_CACHE,
                    f"{artifact_key}.pkl"
                )

                # 캐시 확인
                if os.path.exists(local_cache_path):
                    print("Cache hit.")
                else:
                    print("Computing MI...")
                    train_df = load_train_df(
                        req["mode"],
                        req["fold"],
                        req["seed"],
                        req["cfg"],
                    )
                    mi_dict = _get_mi_helper(
                        train_df,
                        req["seed"],
                        req["n_neighbors"]
                    )

                    with open(local_cache_path, "wb") as f:
                        pickle.dump(mi_dict, f)

                # 응답 업로드
                remote_response = f"{RESPONSES_DIR}/{req['request_id']}.pkl"
                rclone_copy(local_cache_path, remote_response)

                # done 처리
                rclone_move(
                    remote_request,
                    f"{DONE_DIR}/{fname}"
                )

                print("Done.")

        except Exception as e:
            print("Error:", e)

        time.sleep(5)


if __name__ == "__main__":
    main()