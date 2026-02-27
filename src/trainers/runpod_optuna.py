import os
import copy
import optuna
import yaml
import random
import torch
import numpy as np
import argparse
from pathlib import Path
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
    # cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])

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
    # cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])

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
    # cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])

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
    # cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])

def suggest_gin_gru_2_points_params(trial, cfg):
    cfg["model"]["params"]["embedding_dim"] = trial.suggest_categorical("embedding_dim", [16, 32, 64])

    cfg["model"]["params"]["gin_hidden_channel"] = trial.suggest_categorical("gin_hidden_channel", [16, 32, 64, 96])
    cfg["model"]["params"]["gin_layers"] = trial.suggest_int("gin_layers", 1, 6)
    cfg["model"]["params"]["train_eps"] = trial.suggest_categorical("train_eps", [True, False])
    cfg["model"]["params"]["gru_hidden_channel"] = trial.suggest_categorical("gin_hidden_channel", [16, 32, 64, 96])
    cfg["model"]["params"]["dropout_p"] = trial.suggest_float("dropout_p", 0.0, 0.5)
    cfg["model"]["params"]["gin_layer_out_dropout_p"] = trial.suggest_float("gin_layer_out_dropout_p", 0.0, 0.5)
    cfg["model"]["params"]["gru_layer_out_dropout_p"] = trial.suggest_float("gru_layer_out_dropout_p", 0.0, 0.5)

    cfg["edge"]["n_neighbors"] = trial.suggest_categorical("n_neighbors", [1, 3, 5, 7])
    cfg["edge"]["top_k"] = trial.suggest_categorical("top_k", [3, 6, 9, 12])
    cfg["edge"]["threshold"] = trial.suggest_categorical("threshold", [0.0, 0.005, 0.01, 0.02])
    cfg["edge"]["pruning_ratio"] = trial.suggest_categorical("pruning_ratio", [0.0, 0.3, 0.5, 0.7])
   
    cfg["train"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    cfg["train"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    cfg["train"]["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    cfg["train"]["lr_scheduler_patience"] = trial.suggest_categorical("lr_scheduler_patience", [2, 5, 8])
    # cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])


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
    "gin_gru_2_points": suggest_gin_gru_2_points_params,
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
                    cfg=cfg_s,
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

    # ✅ repo 루트(= 이 파일 기준으로 상위로) 잡고 runs/ 절대경로로 고정
    repo_root = Path(__file__).resolve().parents[2]      # 필요하면 parents[2]로 조정
    runs_dir  = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)


    db_path = runs_dir / f"optuna_{model_name}.db"
    storage = f"sqlite:///{db_path.as_posix()}"          # ✅ 절대경로는 sqlite:////... 형태가 됨

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{model_name}",
        storage=storage,
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

# -----------------------
# Utils
# -----------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_runs_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]  # 필요시 조정
    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def _score_from_out(out: dict, model_name: str) -> float:
    if model_name == "xgboost":
        return float(out["roc_auc"])
    return float(out["best_valid_metric"])

# -----------------------
# 핵심: Nested CV + Optuna
# -----------------------
def inner_objective_factory(
    base_cfg: dict,
    root: str,
    outer_fold: int,
    inner_folds: list[int],
    report_metric: str,
    objective_seeds: tuple[int, ...],
):
    """
    inner CV 평균 성능을 maximize 하는 objective.

    중요:
    - 'inner fold'는 반드시 'outer train' 데이터에서만 나뉘어야 합니다.
    - 이 스크립트는 cfg에 outer/inner fold 정보를 심어주고,
      실제 split은 run_single_experiment 내부가 수행한다고 가정합니다.

    run_single_experiment 쪽에서 아래 키들을 읽도록 맞춰주세요:
      cfg["cv"]["outer_fold"], cfg["cv"]["inner_fold"], cfg["cv"]["outer_k"], cfg["cv"]["inner_k"]
      cfg["cv"]["mode"] in {"inner", "outer_eval", "outer_fit"} 등
    """
    def objective(trial: optuna.Trial):
        # trial 고정 seed(재현성)
        trial_seed = 10000 + trial.number
        set_global_seed(trial_seed)

        cfg = copy.deepcopy(base_cfg)
        model_name = cfg["model"]["name"]

        if model_name not in PARAM_SUGGESTORS:
            raise ValueError(f"No suggestor registered for model: {model_name}")
        PARAM_SUGGESTORS[model_name](trial, cfg)

        fold_scores = []
        for inner_fold in inner_folds:
            for seed in objective_seeds:
                cfg_s = copy.deepcopy(cfg)
                cfg_s["train"]["seed"] = int(seed)

                # nested CV 메타정보 주입
                cfg_s.setdefault("cv", {})
                cfg_s["cv"]["outer_fold"] = int(outer_fold)
                cfg_s["cv"]["inner_fold"] = int(inner_fold)
                cfg_s["cv"]["mode"] = "inner"  # inner-CV validate

                try:
                    # MI 캐시도 fold 단위로 분리(권장)
                    mi_cache_path = request_mi(
                        mode="cv",
                        fold=int(inner_fold),
                        seed=int(seed),
                        n_neighbors=cfg_s["edge"]["n_neighbors"],
                        cfg=cfg_s,
                    )

                    out = run_single_experiment(
                        cfg_s,
                        root=root,
                        trial=trial,
                        report_metric=report_metric,
                        edge_cached=True,
                        mi_cache_path=mi_cache_path,
                    )

                    score = _score_from_out(out, model_name)
                    if (score is None) or (not np.isfinite(score)):
                        raise optuna.TrialPruned()

                    fold_scores.append(float(score))

                except optuna.TrialPruned:
                    raise
                except Exception as e:
                    print(f"[Outer {outer_fold} | Inner {inner_fold} | Trial {trial.number}] failed:", repr(e))
                    raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


def run_nested_cv_optuna(
    base_cfg: dict,
    root: str,
    outer_k: int = 5,
    inner_k: int = 3,
    n_trials: int = 50,
    epochs: int = 20,
    report_metric: str = "valid_auc",
    objective_seeds: tuple[int, ...] = (1,),
):
    runs_dir = ensure_runs_dir()
    model_name = base_cfg["model"]["name"]

    # 공용 DB (outer fold별로 study_name만 다르게)
    db_path = runs_dir / f"optuna_nested_{model_name}.db"
    storage = f"sqlite:///{db_path.as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=5,
        max_resource=epochs,
        reduction_factor=3,
    )

    outer_results = []

    for outer_fold in range(outer_k):
        print(f"\n===== [OUTER {outer_fold+1}/{outer_k}] inner optuna start =====")

        # inner folds: 0..inner_k-1 (단, 의미는 'outer train 내부에서의 fold id')
        inner_folds = list(range(inner_k))

        study_name = f"{model_name}__outer={outer_fold}"
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

        objective = inner_objective_factory(
            base_cfg=base_cfg,
            root=root,
            outer_fold=outer_fold,
            inner_folds=inner_folds,
            report_metric=report_metric,
            objective_seeds=objective_seeds,
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_inner = float(study.best_value)

        # inner optuna 결과 저장
        df_path = runs_dir / f"{model_name}__outer={outer_fold}__inner_trials.csv"
        study.trials_dataframe().to_csv(df_path.as_posix(), index=False)

        print(f"[OUTER {outer_fold}] best inner-cv: {best_inner:.6f}")
        print(f"[OUTER {outer_fold}] best params  : {best_params}")

        # -----------------------
        # OUTER 평가(최종 일반화 성능)
        # -----------------------
        print(f"===== [OUTER {outer_fold+1}/{outer_k}] refit + outer-eval =====")

        # best params를 base_cfg에 적용하기 위해:
        # - Optuna가 뱉은 best_params를 "trial 없이" cfg에 세팅하는 함수가 필요.
        #   여기서는 간단히 "고정 Trial"을 만들어 재사용합니다.
        fixed_trial = optuna.trial.FixedTrial(best_params)
        cfg_best = copy.deepcopy(base_cfg)
        if model_name not in PARAM_SUGGESTORS:
            raise ValueError(f"No suggestor registered for model: {model_name}")
        PARAM_SUGGESTORS[model_name](fixed_trial, cfg_best)

        # outer fold 지정
        cfg_best.setdefault("cv", {})
        cfg_best["cv"]["outer_fold"] = int(outer_fold)
        cfg_best["cv"]["mode"] = "outer_eval"  # outer test 평가 모드라고 가정

        # seed 평균(원하면 여러 seed)
        outer_scores = []
        for seed in objective_seeds:
            cfg_s = copy.deepcopy(cfg_best)
            cfg_s["train"]["seed"] = int(seed)

            # outer 평가에서도 MI가 필요하면 fold 기준을 outer_fold로 잡아 캐시 분리
            mi_cache_path = request_mi(
                mode="cv",
                fold=int(outer_fold),
                seed=int(seed),
                n_neighbors=cfg_s["edge"]["n_neighbors"],
                cfg=cfg_s,
            )

            out = run_single_experiment(
                cfg_s,
                root=root,
                trial=None,  # 최종 평가는 optuna trial 불필요
                report_metric=report_metric,
                edge_cached=True,
                mi_cache_path=mi_cache_path,
            )
            score = _score_from_out(out, model_name)
            outer_scores.append(float(score))

        outer_mean = float(np.mean(outer_scores))
        outer_results.append(
            {
                "outer_fold": int(outer_fold),
                "best_inner_cv": best_inner,
                "outer_score": outer_mean,
                "best_params": best_params,
            }
        )

        print(f"[OUTER {outer_fold}] outer_score(mean over seeds): {outer_mean:.6f}")

    # 전체 outer 결과 저장
    out_path = runs_dir / f"{model_name}__nested_cv_summary.yaml"
    with open(out_path.as_posix(), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "model_name": model_name,
                "outer_k": int(outer_k),
                "inner_k": int(inner_k),
                "n_trials": int(n_trials),
                "epochs": int(epochs),
                "report_metric": report_metric,
                "objective_seeds": list(objective_seeds),
                "outer_results": outer_results,
                "outer_score_mean": float(np.mean([r["outer_score"] for r in outer_results])),
                "outer_score_std": float(np.std([r["outer_score"] for r in outer_results])),
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print("\n===== NESTED CV DONE =====")
    print("summary saved to:", out_path.as_posix())
    print("outer mean:", float(np.mean([r["outer_score"] for r in outer_results])))

    return outer_results


if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    root = os.path.join(cur_dir, "..", "data")
    root = os.path.abspath(root)

    args = parse_args()
    base_cfg = load_yaml(args.config)

    # cv 기본 메타 (run_single_experiment에서 읽도록 맞추면 좋음)
    base_cfg.setdefault("cv", {})
    base_cfg["cv"]["outer_k"] = 3# int(args.outer_k)
    base_cfg["cv"]["inner_k"] = 2# int(args.inner_k)

    run_nested_cv_optuna(
        base_cfg=base_cfg,
        root=root,
        outer_k=3,
        inner_k=2,
        n_trials=6,
        epochs=20,
        report_metric="valid_auc",
        objective_seeds=(1,),
    )

'''if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    root = os.path.join(cur_dir, '..', 'data')
    root = os.path.abspath(root)

    args = parse_args()
    base_cfg = load_yaml(args.config)
    run_optuna(base_cfg=base_cfg, root=root, n_trials=50, epochs=20)
'''

    