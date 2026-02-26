import os
import time
import json
import pickle
import subprocess
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif

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
def load_train_df(mode, fold, seed):
    """
    ⚠️ 여기서 실제 fold/split 로직에 맞게
    train subset을 정확히 로드해야 한다.
    """
    # 예시 (수정 필요)
    df = pd.read_pickle("~/data/tedsd_2022_full.pkl")

    if mode == "cv":
        # fold 기준 train subset 생성 로직
        train_df = df[df["fold"] != int(fold)]
    else:
        # seed 기준 random split
        train_df = df.sample(frac=0.8, random_state=seed)

    return train_df


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
                        req["seed"]
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