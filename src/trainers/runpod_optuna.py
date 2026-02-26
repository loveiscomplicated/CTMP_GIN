import os
import copy
import optuna
import yaml
import random
import torch
import numpy as np
import argparse

from src.trainers.run_single_experiment import run_single_experiment  
from scripts.request_mi import request_mi

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True) # config file location
    return p.parse_args()

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def suggest_ctmp_gin_params(trial, cfg):
    cfg["model"]["params"]["embedding_dim"] = trial.suggest_categorical("embedding_dim", [16, 32, 64])
    cfg["model"]["params"]["los_embedding_dim"] = trial.suggest_categorical("los_embedding_dim", [4, 8, 16])

    gin_hidden = trial.suggest_categorical("gin_hidden_channel", [16, 32, 64, 96])
    cfg["model"]["params"]["gin_hidden_channel"] = gin_hidden
    cfg["model"]["params"]["gin_hidden_channel_2"] = gin_hidden

    cfg["model"]["params"]["gin_1_layers"] = trial.suggest_int("gin_1_layers", 1, 3)
    cfg["model"]["params"]["gin_2_layers"] = trial.suggest_int("gin_2_layers", 1, 3)

    cfg["model"]["params"]["dropout_p"] = trial.suggest_float("dropout_p", 0.0, 0.5)
    cfg["model"]["params"]["train_eps"] = trial.suggest_categorical("train_eps", [True, False])
    cfg["model"]["params"]["gate_hidden_ch"] = trial.suggest_categorical("gate_hidden_ch", [None, 64, 128, 256])

    cfg["edge"]["n_neighbors"] = trial.suggest_categorical("n_neighbors", [1, 3, 5, 7])
    cfg["edge"]["top_k"] = trial.suggest_categorical("top_k", [3, 6, 9, 12])
    cfg["edge"]["threshold"] = trial.suggest_categorical("threshold", [0.0, 0.005, 0.01, 0.02])
    cfg["edge"]["pruning_ratio"] = trial.suggest_categorical("pruning_ratio", [0.0, 0.3, 0.5, 0.7])

    cfg["train"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    cfg["train"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    cfg["train"]["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    cfg["train"]["lr_scheduler_patience"] = trial.suggest_categorical("lr_scheduler_patience", [2, 5, 8])
    cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])

def suggest_gin_params(trial, cfg):
    cfg["model"]["params"]["embedding_dim"] = trial.suggest_categorical("embedding_dim", [16, 32, 64])
    cfg["model"]["params"]["gin_dim"] = trial.suggest_categorical("gin_dim", [16, 32, 64, 96])

    cfg["model"]["params"]["gin_layer_num"] = trial.suggest_int("gin_layer_num", 1, 6)

    cfg["model"]["params"]["train_eps"] = trial.suggest_categorical("train_eps", [True, False])

    cfg["edge"]["n_neighbors"] = trial.suggest_categorical("n_neighbors", [1, 3, 5, 7])
    cfg["edge"]["top_k"] = trial.suggest_categorical("top_k", [3, 6, 9, 12])
    cfg["edge"]["threshold"] = trial.suggest_categorical("threshold", [0.0, 0.005, 0.01, 0.02])
    cfg["edge"]["pruning_ratio"] = trial.suggest_categorical("pruning_ratio", [0.0, 0.3, 0.5, 0.7])

    cfg["train"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    cfg["train"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    cfg["train"]["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    cfg["train"]["lr_scheduler_patience"] = trial.suggest_categorical("lr_scheduler_patience", [2, 5, 8])
    cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])

def suggest_a3tgcn_params(trial, cfg):
    cfg["model"]["params"]["embedding_dim"] = trial.suggest_categorical("embedding_dim", [16, 32, 64])
    cfg["model"]["params"]["hidden_channel"] = trial.suggest_categorical("hidden_channel", [16, 32, 64, 96])

    cfg["edge"]["n_neighbors"] = trial.suggest_categorical("n_neighbors", [1, 3, 5, 7])
    cfg["edge"]["top_k"] = trial.suggest_categorical("top_k", [3, 6, 9, 12])
    cfg["edge"]["threshold"] = trial.suggest_categorical("threshold", [0.0, 0.005, 0.01, 0.02])
    cfg["edge"]["pruning_ratio"] = trial.suggest_categorical("pruning_ratio", [0.0, 0.3, 0.5, 0.7])

    cfg["train"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    cfg["train"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    cfg["train"]["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    cfg["train"]["lr_scheduler_patience"] = trial.suggest_categorical("lr_scheduler_patience", [2, 5, 8])
    cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])

def suggest_gin_gru_params(trial, cfg):
    cfg["model"]["params"]["embedding_dim"] = trial.suggest_categorical("embedding_dim", [16, 32, 64])

    cfg["model"]["params"]["gin_hidden_channel"] = trial.suggest_categorical("gin_hidden_channel", [16, 32, 64, 96])
    cfg["model"]["params"]["gin_layers"] = trial.suggest_int("gin_layers", 1, 6)
    cfg["model"]["params"]["train_eps"] = trial.suggest_categorical("train_eps", [True, False])
    cfg["model"]["params"]["gru_hidden_channel"] = trial.suggest_categorical("gin_hidden_channel", [16, 32, 64, 96])
    cfg["model"]["params"]["dropout_p"] = trial.suggest_float("dropout_p", 0.0, 0.5)

    cfg["edge"]["n_neighbors"] = trial.suggest_categorical("n_neighbors", [1, 3, 5, 7])
    cfg["edge"]["top_k"] = trial.suggest_categorical("top_k", [3, 6, 9, 12])
    cfg["edge"]["threshold"] = trial.suggest_categorical("threshold", [0.0, 0.005, 0.01, 0.02])
    cfg["edge"]["pruning_ratio"] = trial.suggest_categorical("pruning_ratio", [0.0, 0.3, 0.5, 0.7])
   
    cfg["train"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    cfg["train"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    cfg["train"]["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    cfg["train"]["lr_scheduler_patience"] = trial.suggest_categorical("lr_scheduler_patience", [2, 5, 8])
    cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])

def suggest_xgboost_params(trial, cfg):
    cfg["train"]["n_estimators"] = trial.suggest_int("n_estimators", 800, 8000, step=200)
    cfg["train"]["max_depth"] = trial.suggest_int("max_depth", 3, 12)
    cfg["train"]["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 20)
    
    cfg["train"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
    cfg["train"]["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
    
    cfg["train"]["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    cfg["train"]["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    cfg["train"]["colsample_bylevel"] = trial.suggest_float("colsample_bylevel", 0.6, 1.0)
    cfg["train"]["colsample_bynode"] = trial.suggest_float("colsample_bynode", 0.6, 1.0)
    
    cfg["train"]["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    cfg["train"]["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 50.0, log=True)

    cfg["train"]["max_leaves"] = trial.suggest_int("max_leaves", 0, 256, step=16)  # 0이면 비활성
    if cfg["train"]["max_leaves"] == 0:
        cfg["train"].pop("max_leaves", None)

PARAM_SUGGESTORS = {
    "ctmp_gin": suggest_ctmp_gin_params,
    "gin": suggest_gin_params,
    "a3tgcn": suggest_a3tgcn_params,
    "gin_gru": suggest_gin_gru_params,
    "xgboost": suggest_xgboost_params
}

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def objective_factory(base_cfg, root, report_metric="valid_auc", objective_seeds=(1,)):
    def objective(trial: optuna.Trial):
        trial_seed = 10000 + trial.number
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        cfg = copy.deepcopy(base_cfg)
        model_name = cfg["model"]["name"]

        if model_name in PARAM_SUGGESTORS:
            PARAM_SUGGESTORS[model_name](trial, cfg)
        else:
            raise ValueError(f"No suggestor registered for model: {model_name}")

        scores = []
        for seed in objective_seeds:
            cfg_s = copy.deepcopy(cfg)
            cfg_s["train"]["seed"] = int(seed)

            try:
                mi_cache_path = request_mi(
                    mode="single",
                    fold=None,
                    seed=seed,
                    n_neighbors=cfg_s["edge"]["n_neighbors"],
                )
                # out = run_single_experiment(cfg_s, root=root, trial=trial, report_metric=report_metric, edge_cached=False)
                out = run_single_experiment(cfg_s, root=root, trial=trial, report_metric=report_metric, edge_cached=True, mi_cache_path=mi_cache_path)
                if model_name == "xgboost":
                    score = float(out["roc_auc"])
                else:
                    score = float(out["best_valid_metric"])

                if (score is None) or (not np.isfinite(score)):
                    raise optuna.TrialPruned()

                scores.append(score)

            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"[Trial {trial.number}] failed:", repr(e))
                raise optuna.TrialPruned()

        return float(sum(scores) / len(scores))
    return objective


def run_optuna(base_cfg, root: str, n_trials: int = 50, epochs: int = 20):
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=5,
        max_resource=epochs,
        reduction_factor=3
    ) # aggressive pruning

    model_name = base_cfg["model"]["name"]
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{model_name}",
        storage=f"sqlite:///runs/optuna_{model_name}.db",
        load_if_exists=True,
    )

    objective = objective_factory(
        base_cfg=base_cfg,
        root=root,
        report_metric="valid_auc",
        objective_seeds=(1,),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("best value:", study.best_value)
    print("best params:", study.best_params)

    # 결과 CSV 저장
    study.trials_dataframe().to_csv(f"runs/{model_name}_optuna_trials.csv", index=False)

    return study


if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    root = os.path.join(cur_dir, '..', 'data')
    root = os.path.abspath(root)
    
    args = parse_args()
    base_cfg = load_yaml(args.config)
    run_optuna(base_cfg=base_cfg, root=root, n_trials=50, epochs=20)


    