from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import ESTIMATOR_VERSION


def _entropy(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True, dropna=False).to_numpy(dtype=float)
    return float(-(probabilities * np.log(probabilities, where=probabilities > 0)).sum())


def _plugin_mi(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y, normalize=True)
    px = table.sum(axis=1).to_numpy(dtype=float)
    py = table.sum(axis=0).to_numpy(dtype=float)
    pxy = table.to_numpy(dtype=float)
    value = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                value += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
    return float(value)


def compute_mi_dict(train_df: pd.DataFrame, score_method: str = "raw_mi") -> dict[str, pd.Series]:
    if score_method not in {"raw_mi", "nmi"}:
        raise ValueError("score_method must be raw_mi or nmi")
    columns = [c for c in train_df.columns if c not in {"REASON", "REASONb"}]
    result: dict[str, pd.Series] = {}
    entropies = {column: _entropy(train_df[column]) for column in columns}
    for source in columns:
        values = {}
        for target in columns:
            if source == target:
                continue
            mi = _plugin_mi(train_df[source], train_df[target])
            if score_method == "nmi":
                denominator = math.sqrt(entropies[source] * entropies[target])
                mi = mi / denominator if denominator > 0 else 0.0
            values[target] = mi
        result[source] = pd.Series(values, dtype=float)
    return result


def node_set_hash(columns: list[str]) -> str:
    return hashlib.blake2b("\0".join(columns).encode("utf-8"), digest_size=12).hexdigest()


def split_fingerprint(train_df: pd.DataFrame) -> str:
    index_hash = pd.util.hash_pandas_object(train_df.index.to_series(), index=False).values
    return hashlib.blake2b(index_hash.tobytes(), digest_size=12).hexdigest()


def mi_cache_path(root: str, train_df: pd.DataFrame, *, score_method: str, seed: int = 42) -> Path:
    columns = [c for c in train_df.columns if c not in {"REASON", "REASONb"}]
    payload = {
        "score_method": score_method,
        "estimator_version": ESTIMATOR_VERSION,
        "node_set_hash": node_set_hash(columns),
        "split_fingerprint": split_fingerprint(train_df),
        "seed": int(seed),
    }
    key = hashlib.blake2b(str(sorted(payload.items())).encode("utf-8"), digest_size=12).hexdigest()
    return Path(root) / "mi" / f"protocol_{key}.pickle"


def load_or_compute_mi(
    root: str,
    train_df: pd.DataFrame,
    *,
    score_method: str,
    seed: int = 42,
) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    path = mi_cache_path(root, train_df, score_method=score_method, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle), {"path": str(path), "cached": True}
    result = compute_mi_dict(train_df, score_method=score_method)
    with path.open("wb") as handle:
        pickle.dump(result, handle)
    return result, {"path": str(path), "cached": False}
