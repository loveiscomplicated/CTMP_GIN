from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .constants import (
    EVAL_FOLDS,
    EVAL_SEEDS,
    EXTERNAL_HPO_RATIO,
    HPO_FOLDS,
    HPO_SEED,
    HPO_SUBSAMPLE_RATIO,
    INNER_VAL_RATIO,
    PROTOCOL_VERSION,
    SPLIT_SEED,
)


def _json_default(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def indices_fingerprint(indices: Iterable[int]) -> str:
    values = np.asarray(list(indices), dtype=np.int64)
    return hashlib.blake2b(values.tobytes(), digest_size=12).hexdigest()


def dataset_fingerprint(labels: np.ndarray) -> str:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    payload = labels.tobytes() + str(labels.shape).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


def _stratified_holdout(indices: np.ndarray, labels: np.ndarray, ratio: float, seed: int):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    train_pos, test_pos = next(splitter.split(np.zeros(len(indices)), labels[indices]))
    return indices[train_pos].astype(np.int64), indices[test_pos].astype(np.int64)


def _folds(indices: np.ndarray, labels: np.ndarray, n_folds: int, seed: int):
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    result = []
    y = labels[indices]
    for fold, (train_pos, val_pos) in enumerate(splitter.split(np.zeros(len(indices)), y)):
        result.append({
            "fold": fold,
            "train_idx": indices[train_pos].astype(np.int64).tolist(),
            "val_idx": indices[val_pos].astype(np.int64).tolist(),
        })
    return result


def create_hpo_artifact(
    labels: np.ndarray,
    path: str | os.PathLike[str],
    *,
    base_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    all_idx = np.arange(len(labels), dtype=np.int64) if base_indices is None else np.asarray(base_indices, dtype=np.int64)
    if len(all_idx) != len(labels):
        raise ValueError("base_indices and labels must have the same length")
    # This subset is generated exactly once and shared by every model/variant.
    subset_pos, _ = next(StratifiedShuffleSplit(
        n_splits=1, test_size=1.0 - HPO_SUBSAMPLE_RATIO, random_state=HPO_SEED
    ).split(np.zeros(len(all_idx)), labels))
    subset_idx = all_idx[subset_pos]
    local_subset_idx = subset_pos.astype(np.int64)
    subset_folds_local = _folds(local_subset_idx, labels, HPO_FOLDS, HPO_SEED)
    subset_folds = []
    for fold_info in subset_folds_local:
        subset_folds.append({
            "fold": fold_info["fold"],
            "train_idx": all_idx[np.asarray(fold_info["train_idx"], dtype=np.int64)].tolist(),
            "val_idx": all_idx[np.asarray(fold_info["val_idx"], dtype=np.int64)].tolist(),
        })
    full_folds_local = _folds(np.arange(len(labels), dtype=np.int64), labels, HPO_FOLDS, HPO_SEED)
    full_folds = []
    for fold_info in full_folds_local:
        full_folds.append({
            "fold": fold_info["fold"],
            "train_idx": all_idx[np.asarray(fold_info["train_idx"], dtype=np.int64)].tolist(),
            "val_idx": all_idx[np.asarray(fold_info["val_idx"], dtype=np.int64)].tolist(),
        })
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "artifact_type": "d_hpo",
        "hpo_seed": HPO_SEED,
        "subset_ratio": HPO_SUBSAMPLE_RATIO,
        "dataset_fingerprint": dataset_fingerprint(labels),
        "d_hpo_idx": all_idx.tolist(),
        "hpo_subset_idx": subset_idx.tolist(),
        # Deliberately independent fold generation over subset and full D_hpo.
        "subset_folds": subset_folds,
        "full_folds": full_folds,
    }
    payload["artifact_fingerprint"] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=12
    ).hexdigest()
    _write_json(path, payload)
    return payload


def create_eval_artifact(
    labels: np.ndarray,
    path: str | os.PathLike[str],
    *,
    eval_seeds: tuple[int, ...] = EVAL_SEEDS,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    all_idx = np.arange(len(labels), dtype=np.int64)
    d_hpo, d_eval = _stratified_holdout(all_idx, labels, EXTERNAL_HPO_RATIO, SPLIT_SEED)
    splits = []
    for eval_seed in eval_seeds:
        skf = StratifiedKFold(n_splits=EVAL_FOLDS, shuffle=True, random_state=eval_seed)
        y_eval = labels[d_eval]
        for fold, (train_pool_pos, test_pos) in enumerate(skf.split(np.zeros(len(d_eval)), y_eval)):
            train_pool = d_eval[train_pool_pos]
            test_idx = d_eval[test_pos]
            train_idx, val_idx = _stratified_holdout(train_pool, labels, INNER_VAL_RATIO, eval_seed)
            splits.append({
                "split_id": f"seed{eval_seed}_fold{fold}",
                "eval_seed": int(eval_seed),
                "fold": int(fold),
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
            })
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "artifact_type": "d_eval",
        "split_seed": SPLIT_SEED,
        "eval_seeds": list(eval_seeds),
        "n_folds": EVAL_FOLDS,
        "inner_val_ratio": INNER_VAL_RATIO,
        "dataset_fingerprint": dataset_fingerprint(labels),
        "d_hpo_idx": d_hpo.tolist(),
        "d_eval_idx": d_eval.tolist(),
        "splits": splits,
    }
    payload["artifact_fingerprint"] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=12
    ).hexdigest()
    _write_json(path, payload)
    return payload


def validate_artifact(payload: dict[str, Any], labels: np.ndarray, kind: str) -> None:
    labels = np.asarray(labels).reshape(-1)
    expected = dataset_fingerprint(labels)
    if payload.get("dataset_fingerprint") != expected:
        raise ValueError(f"{kind} artifact dataset fingerprint does not match current data")
    if kind == "d_hpo":
        if payload.get("hpo_seed") != HPO_SEED or len(payload.get("subset_folds", [])) != HPO_FOLDS:
            raise ValueError("invalid D_hpo artifact seed or fold count")
    elif kind == "d_eval":
        if payload.get("split_seed") != SPLIT_SEED or len(payload.get("splits", [])) != len(EVAL_SEEDS) * EVAL_FOLDS:
            raise ValueError("invalid D_eval artifact seed or split count")
    else:
        raise ValueError(f"unknown artifact kind: {kind}")
