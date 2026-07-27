from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data_processing.data_utils import df_to_tensor, get_col_info, make_binary, organize_labels
from src.data_processing.tackle_missing_value import tackle_missing_value_wrapper


CANONICAL_TEDS_CACHE_VERSION = "canonical_teds_v2"


@dataclass
class CanonicalTEDSBundle:
    processed_df: pd.DataFrame
    encoded_feature_df: pd.DataFrame
    x_tensor: torch.Tensor
    y_tensor: torch.Tensor
    los_raw_tensor: torch.Tensor
    los_encoded_tensor: torch.Tensor
    col_info: tuple[list[str], list[int], list[int], list[int]]
    num_classes: int
    label_col: str
    admission_col_names: list[str]
    admission_col_dims: list[int]
    discharge_target_col_names: list[str]
    discharge_target_col_dims: list[int]
    los_num_classes: int
    raw_row_index: pd.Series
    caseid_series: pd.Series | None

    def schema_metadata(self) -> dict[str, Any]:
        col_list, col_dims, ad_col_index, dis_col_index = self.col_info
        return {
            "schema_version": "canonical_teds_v1",
            "label_col": self.label_col,
            "num_classes": int(self.num_classes),
            "col_list": list(col_list),
            "col_dims": [int(v) for v in col_dims],
            "admission_col_index": [int(v) for v in ad_col_index],
            "discharge_col_index": [int(v) for v in dis_col_index],
            "admission_col_names": list(self.admission_col_names),
            "admission_col_dims": [int(v) for v in self.admission_col_dims],
            "discharge_target_col_names": list(self.discharge_target_col_names),
            "discharge_target_col_dims": [int(v) for v in self.discharge_target_col_dims],
            "los_num_classes": int(self.los_num_classes),
        }


def _file_signature(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {"path": os.path.abspath(path), "exists": False}
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_path(
    root: str,
    *,
    binary: bool,
    ig_label: bool,
    remove_los: bool,
    do_preprocess: bool,
    admission_only: bool,
) -> str:
    raw_data_path = os.path.join(root, "raw", "TEDS_Discharge.csv")
    missing_corrected_path = os.path.join(root, "raw", "missing_corrected.csv")
    source_path = missing_corrected_path if do_preprocess and os.path.exists(missing_corrected_path) else raw_data_path
    payload = {
        "version": CANONICAL_TEDS_CACHE_VERSION,
        "binary": bool(binary),
        "ig_label": bool(ig_label),
        "remove_los": bool(remove_los),
        "do_preprocess": bool(do_preprocess),
        "admission_only": bool(admission_only),
        "source": _file_signature(source_path),
    }
    key = hashlib.blake2b(json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=12).hexdigest()
    cache_dir = os.path.join(root, "processed_cache")
    return os.path.join(cache_dir, f"{CANONICAL_TEDS_CACHE_VERSION}_{key}.pt")


def _load_bundle_cache(cache_path: str) -> CanonicalTEDSBundle | None:
    if not os.path.exists(cache_path):
        return None
    try:
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        print(f"Warning: failed to load dataset cache {cache_path}: {exc}")
        return None


def _save_bundle_cache(cache_path: str, bundle: CanonicalTEDSBundle) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.{os.getpid()}.tmp"
    torch.save(bundle, tmp_path)
    os.replace(tmp_path, cache_path)


def build_canonical_teds_bundle(
    root: str,
    *,
    binary: bool = True,
    ig_label: bool = False,
    remove_los: bool = True,
    do_preprocess: bool = False,
    admission_only: bool = False,
) -> CanonicalTEDSBundle:
    cache_path = _cache_path(
        root,
        binary=binary,
        ig_label=ig_label,
        remove_los=remove_los,
        do_preprocess=do_preprocess,
        admission_only=admission_only,
    )
    cached = _load_bundle_cache(cache_path)
    if cached is not None:
        return cached

    raw_data_path = os.path.join(root, "raw", "TEDS_Discharge.csv")
    missing_corrected_path = os.path.join(root, "raw", "missing_corrected.csv")

    if do_preprocess:
        base_df = tackle_missing_value_wrapper(raw_data_path, missing_corrected_path)
    else:
        base_df = pd.read_csv(raw_data_path)

    raw_row_index = pd.Series(np.arange(len(base_df), dtype=np.int64), name="raw_row_index")
    caseid_series = base_df["CASEID"].copy() if "CASEID" in base_df.columns else None
    base_df = base_df.drop(["DISYR", "CASEID"], axis=1)

    if "REASON" not in base_df.columns:
        raise ValueError("No REASON variable in the raw data.")

    if binary:
        base_df = make_binary(base_df)
        label_col = "REASONb"
    else:
        columns = list(base_df.columns)
        columns.remove("REASON")
        columns.append("REASON")
        base_df = base_df[columns]
        label_col = "REASON"

    if "LOS" not in base_df.columns:
        raise ValueError("No LOS variable in the raw data.")

    encoded_full_df = organize_labels(base_df.copy(), ig_label)
    los_raw_tensor = df_to_tensor(base_df["LOS"])
    los_encoded_tensor = df_to_tensor(encoded_full_df["LOS"])
    y_tensor = df_to_tensor(encoded_full_df[label_col]).unsqueeze(1)
    num_classes = int(encoded_full_df[label_col].nunique())

    feature_df = base_df.copy()
    if remove_los and "LOS" in feature_df.columns:
        feature_df = feature_df.drop("LOS", axis=1)
    if admission_only:
        dis_cols = [c for c in feature_df.columns if c.endswith("_D")]
        feature_df = feature_df.drop(columns=dis_cols)
        if "LOS" in feature_df.columns:
            feature_df = feature_df.drop("LOS", axis=1)
    processed_df = feature_df.copy()

    encoded_feature_df = encoded_full_df.copy()
    if remove_los and "LOS" in encoded_feature_df.columns:
        encoded_feature_df = encoded_feature_df.drop("LOS", axis=1)
    if admission_only:
        dis_cols = [c for c in encoded_feature_df.columns if c.endswith("_D")]
        encoded_feature_df = encoded_feature_df.drop(columns=dis_cols)
        if "LOS" in encoded_feature_df.columns:
            encoded_feature_df = encoded_feature_df.drop("LOS", axis=1)

    if label_col in encoded_feature_df.columns:
        encoded_feature_df = encoded_feature_df.drop(label_col, axis=1)

    if not remove_los and "LOS" in encoded_feature_df.columns:
        los_col = encoded_feature_df.pop("LOS")
        encoded_feature_df["LOS"] = los_col

    col_info = get_col_info(encoded_feature_df, remove_los=remove_los, ig_label=ig_label)
    x_tensor = df_to_tensor(encoded_feature_df.drop("LOS", axis=1, errors="ignore"))

    col_list, col_dims, ad_col_index, _ = col_info
    admission_col_names = [str(col_list[idx]) for idx in ad_col_index]
    admission_col_dims = [int(col_dims[idx]) for idx in ad_col_index]
    discharge_target_col_names = [c for c in encoded_feature_df.columns if c.endswith("_D")]
    discharge_target_col_dims = [int(encoded_feature_df[c].nunique()) for c in discharge_target_col_names]
    los_num_classes = int(encoded_full_df["LOS"].nunique())

    bundle = CanonicalTEDSBundle(
        processed_df=processed_df,
        encoded_feature_df=encoded_feature_df,
        x_tensor=x_tensor,
        y_tensor=y_tensor,
        los_raw_tensor=los_raw_tensor,
        los_encoded_tensor=los_encoded_tensor,
        col_info=col_info,
        num_classes=num_classes,
        label_col=label_col,
        admission_col_names=admission_col_names,
        admission_col_dims=admission_col_dims,
        discharge_target_col_names=discharge_target_col_names,
        discharge_target_col_dims=discharge_target_col_dims,
        los_num_classes=los_num_classes,
        raw_row_index=raw_row_index,
        caseid_series=caseid_series,
    )
    _save_bundle_cache(cache_path, bundle)
    return bundle
