from __future__ import annotations

import copy
from typing import Any

try:
    import optuna
except ImportError:  # type annotations remain deferred; HPO stages check dependency at runtime.
    optuna = None  # type: ignore[assignment]


def suggest_protocol_params(trial: optuna.Trial, cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply the protocol HPO space and return the mutated config."""
    cfg = copy.deepcopy(cfg)
    model = cfg["model"]["name"]
    params = cfg.setdefault("model", {}).setdefault("params", {})
    train = cfg.setdefault("train", {})
    edge = cfg.setdefault("edge", {})

    params["embedding_dim"] = trial.suggest_categorical("embedding_dim", [16, 32, 64])
    params["dropout_p"] = trial.suggest_float("dropout_p", 0.0, 0.5)
    train["batch_size"] = trial.suggest_categorical("batch_size", [256, 512, 1024])
    train["learning_rate"] = trial.suggest_float("learning_rate", 2e-4, 8e-3, log=True)
    train["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    train["optimizer"] = "adamw"
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
    elif model == "xgboost":
        # XGBoost does not consume the neural common space.
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


def apply_trial_params(cfg: dict[str, Any], params: dict[str, Any], graph_config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    model = cfg["model"]["name"]
    model_params = cfg.setdefault("model", {}).setdefault("params", {})
    train = cfg.setdefault("train", {})
    for key, value in params.items():
        if key in {"batch_size", "learning_rate", "weight_decay", "n_estimators", "max_depth", "min_child_weight", "gamma", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"}:
            train[key] = value
        else:
            model_params[key] = value
    train["optimizer"] = "adamw"
    train["lr_scheduler_patience"] = 5
    train["early_stopping_patience"] = 15
    model_params["train_eps"] = True
    if model == "ctmp_gin":
        model_params["gate_hidden_ch"] = None
    cfg["edge"] = {**cfg.get("edge", {}), **{
        "is_mi_based": True,
        "score_method": graph_config["score_method"],
        "threshold": graph_config["threshold"],
        "top_k": graph_config["top_k"],
        "pruning_ratio": graph_config["pruning_ratio"],
    }}
    train["eval_drop_last"] = False
    train["drop_last"] = True
    return cfg
