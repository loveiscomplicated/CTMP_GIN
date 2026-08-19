from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

try:
    import optuna
except ImportError:  # type annotations remain deferred; HPO stages check dependency at runtime.
    optuna = None  # type: ignore[assignment]


RAW_MI_THRESHOLDS = (0.0, 0.005, 0.01, 0.02)
NMI_THRESHOLDS = (0.0, 0.02, 0.05, 0.10)
TOP_K_CHOICES = (3, 6, 9, 12)
PRUNING_RATIO_CHOICES = (0.0, 0.3, 0.5, 0.7)

TRAIN_PARAM_KEYS = {
    "batch_size",
    "learning_rate",
    "weight_decay",
    "n_estimators",
    "max_depth",
    "min_child_weight",
    "gamma",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
}
GRAPH_PARAM_KEYS = {
    "score_method",
    "threshold",
    "threshold_raw_mi",
    "threshold_nmi",
    "top_k",
    "pruning_ratio",
}


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.blake2b(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"),
        digest_size=12,
    ).hexdigest()


def normalize_graph_params(params: dict[str, Any], graph_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the effective MI graph config represented by trial params."""
    source = params if graph_config is None else {**params, **graph_config}
    score_method = str(source.get("score_method", "raw_mi"))
    if score_method not in {"raw_mi", "nmi"}:
        raise ValueError("score_method must be raw_mi or nmi")

    if "threshold" in source:
        threshold = source["threshold"]
    elif score_method == "raw_mi":
        threshold = source.get("threshold_raw_mi", RAW_MI_THRESHOLDS[0])
    else:
        threshold = source.get("threshold_nmi", NMI_THRESHOLDS[0])

    return {
        "is_mi_based": True,
        "score_method": score_method,
        "threshold": float(threshold),
        "top_k": int(source.get("top_k", TOP_K_CHOICES[1])),
        "pruning_ratio": float(source.get("pruning_ratio", PRUNING_RATIO_CHOICES[2])),
    }


def selected_config_fingerprint(
    *,
    model_name: str,
    variant: str,
    params: dict[str, Any],
    graph_params: dict[str, Any],
) -> str:
    return _fingerprint({
        "model_name": model_name,
        "variant": variant,
        "params": params,
        "graph_params": graph_params,
    })


def suggest_protocol_params(trial: optuna.Trial, cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply the protocol HPO space and return the mutated config."""
    cfg = copy.deepcopy(cfg)
    model = cfg["model"]["name"]
    params = cfg.setdefault("model", {}).setdefault("params", {})
    train = cfg.setdefault("train", {})
    edge = cfg.setdefault("edge", {})

    if model == "xgboost":
        train["n_estimators"] = trial.suggest_int("n_estimators", 800, 8000, step=200)
        train["max_depth"] = trial.suggest_int("max_depth", 3, 12)
        train["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 20)
        train["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
        train["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
        train["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        train["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        train["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
        train["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 50.0, log=True)
        edge["is_mi_based"] = True
        return cfg

    score_method = trial.suggest_categorical("score_method", ["raw_mi", "nmi"])
    edge["score_method"] = score_method
    if score_method == "raw_mi":
        edge["threshold"] = trial.suggest_categorical("threshold_raw_mi", list(RAW_MI_THRESHOLDS))
    else:
        edge["threshold"] = trial.suggest_categorical("threshold_nmi", list(NMI_THRESHOLDS))
    edge["top_k"] = trial.suggest_categorical("top_k", list(TOP_K_CHOICES))
    edge["pruning_ratio"] = trial.suggest_categorical("pruning_ratio", list(PRUNING_RATIO_CHOICES))
    edge["is_mi_based"] = True

    params["embedding_dim"] = trial.suggest_categorical("embedding_dim", [16, 32, 64])
    params["dropout_p"] = trial.suggest_float("dropout_p", 0.0, 0.5)
    train["batch_size"] = trial.suggest_categorical("batch_size", [256, 512, 1024])
    train["learning_rate"] = trial.suggest_float("learning_rate", 2e-4, 8e-3, log=True)
    train["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    train["optimizer"] = "adam"
    train["lr_scheduler_patience"] = 5
    train["early_stopping_patience"] = 15
    params["train_eps"] = True
    params["gate_hidden_ch"] = None

    if model == "ctmp_gin":
        params["los_embedding_dim"] = trial.suggest_categorical("los_embedding_dim", [4, 8, 16])
        params["gin_hidden_channel"] = trial.suggest_categorical("gin_hidden_channel", [16, 32, 64, 96])
        params["gin_hidden_channel_2"] = trial.suggest_categorical("gin_hidden_channel_2", [16, 32, 64, 96])
        params["gin_1_layers"] = trial.suggest_int("gin_1_layers", 1, 3)
        params["gin_2_layers"] = trial.suggest_int("gin_2_layers", 1, 3)
    elif model == "gin":
        params["gin_dim"] = trial.suggest_categorical("gin_dim", [16, 32, 64, 96])
        params["gin_layer_num"] = trial.suggest_int("gin_layer_num", 1, 6)
    elif model in {"gin_gru", "gin_gru_2_points"}:
        params["gin_hidden_channel"] = trial.suggest_categorical("gin_hidden_channel", [16, 32, 64, 96])
        params["gin_layers"] = trial.suggest_int("gin_layers", 1, 4)
        params["gru_hidden_channel"] = trial.suggest_categorical("gru_hidden_channel", [16, 32, 64, 96])
    elif model in {"a3tgcn", "a3tgcn_2_points"}:
        params["hidden_channel"] = trial.suggest_categorical("hidden_channel", [16, 32, 64, 96])
        params["num_layers"] = trial.suggest_int("num_layers", 1, 3)

    return cfg


def apply_trial_params(
    cfg: dict[str, Any],
    params: dict[str, Any],
    graph_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    model = cfg["model"]["name"]
    model_params = cfg.setdefault("model", {}).setdefault("params", {})
    train = cfg.setdefault("train", {})
    for key, value in params.items():
        if key in TRAIN_PARAM_KEYS:
            train[key] = value
        elif key in GRAPH_PARAM_KEYS:
            continue
        else:
            model_params[key] = value
    if model != "xgboost":
        train["optimizer"] = "adam"
        train["lr_scheduler_patience"] = 5
        train["early_stopping_patience"] = 15
        model_params["train_eps"] = True
    if model == "ctmp_gin":
        model_params["gate_hidden_ch"] = None
    edge = cfg.setdefault("edge", {})
    if edge.get("is_mi_based", True):
        edge.update(normalize_graph_params(params, graph_config))
    train["eval_drop_last"] = False
    train["drop_last"] = True
    return cfg
