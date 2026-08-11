from __future__ import annotations

import copy
from typing import Any

import numpy as np


VARIANTS = {
    "full": {"tier": "reference", "hpo": True},
    "A1": {"name": "w/o CT-edge", "hpo": True},
    "A2": {"name": "w/o LOS-edge", "hpo": True},
    "A3": {"name": "LOS-as-node", "hpo": True},
    "A4": {"name": "LOS-shuffled", "hpo": False, "repetitions": 5},
    "C1": {"name": "GIN admission-only", "hpo": True, "source": "gin"},
    "C3": {"name": "GIN discharge-only", "hpo": True, "source": "gin"},
    "xgboost_admission": {"name": "XGBoost admission-only", "hpo": True, "source": "xgboost"},
    "B1": {"name": "bidirectional CT-edge", "hpo": False},
    "B3": {"name": "w/o merged stream", "hpo": False},
    "w/o_merged_stream": {"name": "w/o merged stream", "hpo": False},
    "w/o_gated_fusion": {"name": "w/o GatedFusion", "hpo": False},
    "w/o_mi_edge": {"name": "w/o MI edge", "hpo": False},
    "w/o_preprocessing": {"name": "w/o preprocessing", "hpo": False},
}


def apply_variant(cfg: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown protocol variant: {variant}")
    result = copy.deepcopy(cfg)
    model = result.setdefault("model", {}).setdefault("params", {})
    if variant == "A1":
        model["ct_edge_mode"] = "none"
    elif variant == "A2":
        model["inter_edge_feature_mode"] = "zero"
        model["remove_los_edge"] = True
    elif variant == "A3":
        result["los_as_node"] = True
        model["inter_edge_feature_mode"] = "zero"
        model["remove_los_edge"] = True
    elif variant == "A4":
        result.setdefault("evaluation", {})["los_shuffle_repetitions"] = int(VARIANTS["A4"]["repetitions"])
        result.setdefault("evaluation", {})["use_los_shuffle_as_primary"] = True
    elif variant == "C1" or variant == "xgboost_admission":
        result["admission_only"] = True
    elif variant == "C3":
        result["discharge_only"] = True
    elif variant == "B1":
        model["ct_edge_mode"] = "bidirectional"
    elif variant in {"B3", "w/o_merged_stream"}:
        model["fusion_stream_mask"] = ["ad", "dis"]
    elif variant == "w/o_gated_fusion":
        model["remove_gated_fusion"] = True
    elif variant == "w/o_mi_edge":
        result.setdefault("edge", {})["is_mi_based"] = False
    elif variant == "w/o_preprocessing":
        result.setdefault("train", {})["do_preprocess"] = False
    return result


def shuffle_los_for_evaluation(los: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(los).copy()
    np.random.default_rng(seed).shuffle(values)
    return values


def edge_set_jaccard(first: set[tuple[int, int]], second: set[tuple[int, int]]) -> float:
    union = first | second
    return 1.0 if not union else len(first & second) / len(union)
