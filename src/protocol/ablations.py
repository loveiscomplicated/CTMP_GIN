from __future__ import annotations

import copy
from typing import Any

import numpy as np


VARIANTS = {
    "full": {"tier": "reference", "hpo": True},
    "A1": {"name": "w/o CT-edge", "hpo": False},
    "A2": {"name": "w/o LOS-edge", "hpo": False},
    "A3": {"name": "LOS-as-node", "hpo": False},
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


ABLATION_ALLOWED_DIFFS = {
    "A1": {"model.params.ct_edge_mode"},
    "A2": {"model.params.inter_edge_feature_mode", "model.params.remove_los_edge"},
    "A3": {"los_as_node", "model.params.inter_edge_feature_mode", "model.params.remove_los_edge"},
    "A4": {"evaluation.los_shuffle_repetitions", "evaluation.use_los_shuffle_as_primary"},
    "B1": {"model.params.ct_edge_mode"},
    "B3": {"model.params.fusion_stream_mask"},
    "w/o_merged_stream": {"model.params.fusion_stream_mask"},
    "w/o_gated_fusion": {"model.params.remove_gated_fusion"},
    "w/o_mi_edge": {"edge.is_mi_based"},
    "w/o_preprocessing": {"train.do_preprocess"},
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


def _flatten_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        diffs = []
        for key in sorted(set(before) | set(after)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in before:
                if isinstance(after[key], dict):
                    diffs.extend(_flatten_diff({}, after[key], child_path))
                else:
                    diffs.append({"path": child_path, "before": None, "after": after[key]})
            elif key not in after:
                if isinstance(before[key], dict):
                    diffs.extend(_flatten_diff(before[key], {}, child_path))
                else:
                    diffs.append({"path": child_path, "before": before[key], "after": None})
            else:
                diffs.extend(_flatten_diff(before[key], after[key], child_path))
        return diffs
    if before != after:
        return [{"path": path, "before": before, "after": after}]
    return []


def validate_ablation_mutation(
    parent_cfg: dict[str, Any],
    effective_cfg: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    if variant not in ABLATION_ALLOWED_DIFFS:
        raise ValueError(f"{variant} has no controlled-ablation diff whitelist")
    diffs = _flatten_diff(parent_cfg, effective_cfg)
    allowed = ABLATION_ALLOWED_DIFFS[variant]
    unexpected = [item for item in diffs if item["path"] not in allowed]
    if unexpected:
        raise ValueError(f"{variant} changed non-whitelisted config paths: {unexpected}")
    return {
        "variant": variant,
        "allowed_paths": sorted(allowed),
        "diffs": diffs,
    }


def shuffle_los_for_evaluation(los: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(los).copy()
    np.random.default_rng(seed).shuffle(values)
    return values


def edge_set_jaccard(first: set[tuple[int, int]], second: set[tuple[int, int]]) -> float:
    union = first | second
    return 1.0 if not union else len(first & second) / len(union)
