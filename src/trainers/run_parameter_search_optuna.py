import os
import copy
import optuna
import yaml
import random
import torch
import numpy as np

from src.trainers.run_single_experiment import run_single_experiment  # 네 경로에 맞게 수정

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

        if cfg["model"]["name"] == "ctmp_gin":
            # ---- suggest hyperparams ----
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

        elif cfg["model"]["name"] == "gin":
            # ---- suggest hyperparams ----
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
            ...
        else:
            raise ValueError("no matching model name (ctmp_gin or gin)")
        
        scores = []
        for seed in objective_seeds:
            cfg_s = copy.deepcopy(cfg)
            cfg_s["train"]["seed"] = int(seed)

            try:
                out = run_single_experiment(cfg_s, root=root, trial=trial, report_metric=report_metric, edge_cached=False)
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


def run_optuna(config_path: str, root: str, n_trials: int = 50):

    os.makedirs("runs", exist_ok=True)

    config_path = os.path.abspath(config_path)
    root = os.path.abspath(root)

    base_cfg = load_cfg(config_path)

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=5,
        max_resource=base_cfg["train"]["epochs"],
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
    
    config_path = os.path.join(cur_dir, '..', '..', 'configs', 'gin.yaml')
    root = os.path.join(cur_dir, '..', 'data')
    
    config_path = os.path.abspath(config_path)
    root = os.path.abspath(root)

    run_optuna(config_path=config_path, root=root, n_trials=50)
