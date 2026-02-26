import os
import json
import time
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path


DATASET_ID = "tedsd_2022"
REMOTE_BASE = "gdrive:CTMP_GIN_mi_service"
LOCAL_CACHE_DIR = Path("/workspace/CTMP_GIN/cache/mi_dict")
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str]) -> str:
    """Run command and return stdout."""
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return p.stdout


def _artifact_key(mode: str, fold: int | None, seed: int, n_neighbors: int) -> str:
    if mode == "cv":
        if fold is None:
            raise ValueError("mode=cv requires fold")
        return (
            f"mi__ds={DATASET_ID}__mode=cv__fold={fold}__seed={seed}__nn={n_neighbors}__disc=1"
        )
    if mode == "single":
        return f"mi__ds={DATASET_ID}__mode=single__seed={seed}__nn={n_neighbors}__disc=1"
    raise ValueError("mode must be 'cv' or 'single'")


def _request_id_from_artifact(artifact_key: str) -> str:
    # 너무 긴 파일명을 피하려고 hash를 일부 섞음 (원하면 artifact_key 그대로 써도 됨)
    h = hashlib.sha1(artifact_key.encode("utf-8")).hexdigest()[:10]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}__{h}"


def request_mi(
    *,
    mode: str, # single or cv
    fold: int | None,
    seed: int,
    cfg,
    n_neighbors: int,
    poll_interval_sec: int = 3,
    timeout_sec: int | None = None,
) -> str:
    """
    Request mi_dict from Mac worker via rclone remote folder, blocking until ready.
    Returns local path to the cached mi_dict pickle.
    """
    artifact_key = _artifact_key(mode, fold, seed, n_neighbors)
    local_pkl = LOCAL_CACHE_DIR / f"{artifact_key}.pkl"

    # 1) local cache hit
    if local_pkl.exists():
        return str(local_pkl)

    # 2) local lock: avoid duplicate requests per node
    lock_dir = LOCAL_CACHE_DIR / f"{artifact_key}.lock"
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            # someone else is requesting; wait for file
            if local_pkl.exists():
                return str(local_pkl)
            time.sleep(1)

    try:
        # lock 획득 후에도 누군가 이미 받아놨을 수 있음
        if local_pkl.exists():
            return str(local_pkl)

        request_id = _request_id_from_artifact(artifact_key)
        tmp_json = Path("/tmp") / f"{request_id}.json"

        payload = {
            "request_id": request_id,
            "artifact_key": artifact_key,
            "mode": mode,
            "fold": (fold if fold is not None else "none"),
            "seed": seed,
            "n_neighbors": n_neighbors,
            "cfg": cfg,
        }
        tmp_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        # 3) upload request
        _run(["rclone", "copyto", str(tmp_json), f"{REMOTE_BASE}/requests/{request_id}.json"])

        # 4) poll for response
        start = time.time()
        remote_pkl = f"{REMOTE_BASE}/responses/{request_id}.pkl"
        while True:
            # timeout
            if timeout_sec is not None and (time.time() - start) > timeout_sec:
                raise TimeoutError(f"Timed out waiting for {remote_pkl}")

            # check exists by attempting lsf on exact file path (simpler/cheap)
            try:
                _run(["rclone", "lsf", remote_pkl])
                break
            except subprocess.CalledProcessError:
                time.sleep(poll_interval_sec)

        # 5) download to local cache
        _run(["rclone", "copyto", remote_pkl, str(local_pkl)])

        return str(local_pkl)

    finally:
        # release lock
        try:
            lock_dir.rmdir()
        except OSError:
            # lock_dir not empty or other issue; ignore
            pass

