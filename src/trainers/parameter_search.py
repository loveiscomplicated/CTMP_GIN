from __future__ import annotations
import random
import copy
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import optuna
import yaml

from src.trainers.run_single_experiment import run_single_experiment

# -----------------------------
# Utils
# -----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dict."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

# -----------------------------
# 1) Trial -> config override
# -----------------------------

def suggest_hparams(trial: optuna.Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return an override dict shaped like the YAML:
      {"edge": {...}, "model": {"params": {...}}, "train": {...}}
    """
    # ---- model ----
    embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64])
    los_embedding_dim = trial.suggest_categorical("los_embedding_dim", [4, 8, 16])

    gin_hidden_channel = trial.suggest_categorical("gin_hidden_channel", [16, 32, 64, 96])
    # mismatch 방지: 동일하게 고정 권장
    gin_hidden_channel_2 = gin_hidden_channel

    gin_1_layers = trial.suggest_int("gin_1_layers", 1, 3)
    gin_2_layers = trial.suggest_int("gin_2_layers", 1, 3)

    dropout_p = trial.suggest_float("dropout_p", 0.0, 0.5)

    train_eps = trial.suggest_categorical("train_eps", [True, False])

    # gate_hidden_ch: None 또는 특정 값
    gate_hidden_ch = trial.suggest_categorical("gate_hidden_ch", [None, 64, 128, 256])

    # ---- edge ----
    n_neighbors = trial.suggest_categorical("n_neighbors", [1, 3, 5, 7])
    top_k = trial.suggest_categorical("top_k", [3, 6, 9, 12])
    threshold = trial.suggest_categorical("threshold", [0.0, 0.005, 0.01, 0.02])
    pruning_ratio = trial.suggest_categorical("pruning_ratio", [0.0, 0.3, 0.5, 0.7])

    # ---- train/optim ----
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)

    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])

    lr_scheduler_patience = trial.suggest_categorical("lr_scheduler_patience", [2, 5, 8])
    early_stopping_patience = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])

    # decision_threshold는 objective에는 보통 안 넣는 걸 추천.
    # 필요하면 "최종 best만" threshold sweep으로 결정.

    override = {
        "edge": {
            "n_neighbors": n_neighbors,
            "top_k": top_k,
            "threshold": threshold,
            "pruning_ratio": pruning_ratio,
            # base_cfg의 is_mi_based / return_edge_attr 등은 그대로 둠
        },
        "model": {
            "params": {
                "embedding_dim": embedding_dim,
                "los_embedding_dim": los_embedding_dim,
                "gin_hidden_channel": gin_hidden_channel,
                "gin_hidden_channel_2": gin_hidden_channel_2,
                "gin_1_layers": gin_1_layers,
                "gin_2_layers": gin_2_layers,
                "dropout_p": dropout_p,
                "train_eps": train_eps,
                "gate_hidden_ch": gate_hidden_ch,
            }
        },
        "train": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lr_scheduler_patience": lr_scheduler_patience,
            "early_stopping_patience": early_stopping_patience,
            # 아래 2개는 YAML에 없으니 네 학습 코드가 받게 추가해야 함(추천)
            "weight_decay": weight_decay,
            "optimizer": optimizer,
        }
    }
    return override

# -----------------------------
# 2) Run one training and return val metric
# -----------------------------

def train_eval_once(cfg: Dict[str, Any], trial: Optional[optuna.Trial] = None) -> float:
    """
    Must return a single float (higher is better).
    You should implement this by calling your existing pipeline.

    Recommended: return val_auc (threshold-free).
    """
    # TODO: 아래를 너 코드에 맞게 연결
    # 예:
    #   set_seed(cfg["train"]["seed"])
    #   device = resolve_device(cfg["device"])
    #   dataset = load_dataset(...)
    #   edge_index = build_edge(dataset, cfg["edge"])
    #   model = build_model(cfg["model"])
    #   metrics = run_train_loop(model, dataset, edge_index, cfg["train"], trial=trial)
    #   return metrics["val_auc"]

    raise NotImplementedError("Connect train_eval_once() to your training pipeline.")

# -----------------------------
# 3) (Option) Multi-seed evaluation
# -----------------------------

def train_eval_multi_seed(cfg: Dict[str, Any], seeds: List[int], trial: Optional[optuna.Trial] = None) -> float:
    scores = []
    for s in seeds:
        cfg_s = copy.deepcopy(cfg)
        cfg_s["train"]["seed"] = int(s)
        score = train_eval_once(cfg_s, trial=trial)
        scores.append(score)
    return float(np.mean(scores))

# -----------------------------
# 4) Objective
# -----------------------------

def make_objective(base_cfg: Dict[str, Any], seeds_for_objective: List[int]):
    def objective(trial: optuna.Trial) -> float:
        # trial seed 고정(재현성)
        trial_seed = 10_000 + trial.number
        set_global_seed(trial_seed)

        cfg = copy.deepcopy(base_cfg)
        override = suggest_hparams(trial, base_cfg)
        cfg = deep_update(cfg, override)

        # 안정성 체크(필요시 더 추가)
        if cfg["model"]["params"]["gin_hidden_channel_2"] != cfg["model"]["params"]["gin_hidden_channel"]:
            raise optuna.TrialPruned()

        # 핵심: 여기서 기존 학습 루프 호출
        score = train_eval_multi_seed(cfg, seeds_for_objective, trial=trial)

        # Optuna에 기록(나중에 분석 편함)
        trial.set_user_attr("cfg_snapshot", cfg)

        return score
    return objective

# -----------------------------
# 5) Study runner
# -----------------------------

def run_optuna(
    config_path: str,
    study_dir: str = "runs/optuna_ctmp_gin",
    n_trials: int = 50,
    objective_seeds: List[int] = [1],  # 1개로 빠르게 찾고, 최종에서 multi-seed 재평가 추천
    direction: str = "maximize",
):
    base_cfg = load_yaml(config_path)

    Path(study_dir).mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    objective = make_objective(base_cfg, objective_seeds)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save results
    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }

    with open(Path(study_dir) / "best.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    # best cfg snapshot 저장 (trial user_attr 사용)
    best_cfg = study.best_trial.user_attrs.get("cfg_snapshot", None)
    if best_cfg is not None:
        save_yaml(best_cfg, Path(study_dir) / "best_config.yaml")

    # 전체 trials CSV 저장
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(Path(study_dir) / "trials.csv", index=False)

    print("DONE")
    print(best)
    return study

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--study_dir", type=str, default="runs/optuna_ctmp_gin")
    p.add_argument("--n_trials", type=int, default=50)
    args = p.parse_args()

    run_optuna(
        config_path=args.config,
        study_dir=args.study_dir,
        n_trials=args.n_trials,
        objective_seeds=[1],
    )
