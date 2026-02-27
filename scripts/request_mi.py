import json
import time
import hashlib
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


DATASET_ID = "tedsd_2022"
REMOTE_BASE = "gdrive:CTMP_GIN_mi_service"

LOCAL_CACHE_DIR = Path("/workspace/CTMP_GIN/cache/mi_dict")
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str]) -> str:
    """
    Run command and return stdout.
    Raise RuntimeError with stdout/stderr on failure.
    """
    try:
        p = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return p.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"[CMD FAILED]\n"
            f"cmd: {' '.join(cmd)}\n"
            f"returncode: {e.returncode}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}\n"
        ) from e


def _artifact_key(mode: str, fold: int | None, seed: int, n_neighbors: int, model_name) -> str:
    if mode == "cv":
        if fold is None:
            raise ValueError("mode=cv requires fold")
        return f"mi__ds={DATASET_ID}__mode=cv__fold={fold}__seed={seed}__nn={n_neighbors}__model={model_name}__disc=1"
    if mode == "single":
        return f"mi__ds={DATASET_ID}__mode=single__seed={seed}__nn={n_neighbors}__model={model_name}__disc=1"
    raise ValueError("mode must be 'cv' or 'single'")


def _request_id_from_artifact(artifact_key: str) -> str:
    # short stable hash + timestamp (avoid overly long names, avoid collisions)
    h = hashlib.sha1(artifact_key.encode("utf-8")).hexdigest()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}__{h}"


def _ensure_remote_dirs() -> None:
    # idempotent
    _run(["rclone", "mkdir", f"{REMOTE_BASE}/requests"])
    _run(["rclone", "mkdir", f"{REMOTE_BASE}/responses"])


def _acquire_lock(lock_dir: Path, local_pkl: Path) -> bool:
    """
    Acquire local lock directory. Returns True if acquired.
    If lock exists, waits until either pkl exists (then returns False) or lock becomes available.
    """
    while True:
        if local_pkl.exists():
            return False
        try:
            lock_dir.mkdir()
            return True
        except FileExistsError:
            # someone else is requesting
            if local_pkl.exists():
                return False
            time.sleep(1)


def _release_lock(lock_dir: Path) -> None:
    # robust cleanup
    shutil.rmtree(lock_dir, ignore_errors=True)


def request_mi(
    *,
    model_name: str,
    mode: str,  # "single" or "cv"
    fold: int | None,
    seed: int,
    cfg: Any,
    n_neighbors: int,
    poll_interval_sec: int = 3,
    timeout_sec: int | None = None,
    serialize_cfg_default_str: bool = True,
    verbose_poll: bool = False,
    use_cache: bool = True,
) -> str:
    """
    Request mi_dict from worker via rclone remote folder, blocking until ready.
    Returns local path to the cached mi_dict pickle.

    Remote protocol:
      - Upload:  {REMOTE_BASE}/requests/{request_id}.json
      - Worker writes: {REMOTE_BASE}/responses/{request_id}.pkl
      - Download to: {LOCAL_CACHE_DIR}/{artifact_key}.pkl

    Args:
        mode: "single" or "cv"
        fold: required if mode=="cv"
        seed: random seed
        cfg: config payload (should be JSON-serializable)
        n_neighbors: MI estimator neighbors
        poll_interval_sec: seconds between polls
        timeout_sec: None for no timeout
        serialize_cfg_default_str: if True, json.dumps(..., default=str) to avoid non-serializable objects
        verbose_poll: if True, prints progress every ~10 polls
        use_cache: if True, use local cache if available

    Returns:
        str: local pickle path
    """
    artifact_key = _artifact_key(mode, fold, seed, n_neighbors, model_name)
    local_pkl = LOCAL_CACHE_DIR / f"{artifact_key}.pkl"

    # 1) local cache hit
    if use_cache and local_pkl.exists():
        return str(local_pkl)

    # 2) local lock (avoid duplicate request on same node)
    lock_dir = LOCAL_CACHE_DIR / f"{artifact_key}.lock"
    acquired = _acquire_lock(lock_dir, local_pkl)
    if not acquired:
        # someone else produced it while we waited
        return str(local_pkl)

    try:
        # re-check after lock
        if use_cache and local_pkl.exists():
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
            "use_cache": use_cache,
        }

        dumps_kwargs = {"ensure_ascii": False}
        if serialize_cfg_default_str:
            dumps_kwargs["default"] = str

        tmp_json.write_text(json.dumps(payload, **dumps_kwargs), encoding="utf-8")

        # 3) ensure remote dirs exist
        _ensure_remote_dirs()

        # 4) upload request
        remote_json = f"{REMOTE_BASE}/requests/{request_id}.json"
        _run(["rclone", "copyto", str(tmp_json), remote_json])

        # 5) poll for response
        start = time.time()
        remote_pkl = f"{REMOTE_BASE}/responses/{request_id}.pkl"

        polls = 0
        while True:
            if timeout_sec is not None and (time.time() - start) > timeout_sec:
                raise TimeoutError(f"Timed out waiting for {remote_pkl}")

            try:
                # Existence check: success(returncode 0) is enough
                _run(["rclone", "lsf", remote_pkl])
                break
            except RuntimeError:
                polls += 1
                if verbose_poll and polls % 10 == 0:
                    elapsed = int(time.time() - start)
                    print(f"[request_mi] waiting... elapsed={elapsed}s remote={remote_pkl}")
                time.sleep(poll_interval_sec)

        # 6) download to local cache (atomic-ish)
        tmp_local = local_pkl.with_suffix(".pkl.tmp")
        _run(["rclone", "copyto", remote_pkl, str(tmp_local)])
        tmp_local.replace(local_pkl)

        return str(local_pkl)

    finally:
        _release_lock(lock_dir)