import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root.parent))

import yaml
import argparse
from src.trainers.run_single_experiment import run_single_experiment
from src.trainers.run_kfold_cv import run_kfold_experiment
from scripts.request_mi import request_mi

cur_dir = os.path.dirname(__file__)
root = os.path.join(cur_dir, 'data')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True) # config file location
    # overrides
    # p.add_argument("--model", type=str, default=None) no need, model selection only based on config
    # more detailed adjustment able in config file
    p.add_argument("--is_mi_based_edge", type=int, default=None)
    p.add_argument("--edge_cache_path", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--decision_threshold", type=float, default=None)
    p.add_argument("--binary", type=int, default=None)
    p.add_argument("--cv", type=bool, default=True)
    return p.parse_args()

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def override_cfg(cfg: dict, args) -> dict:
    if args.device is not None:
        cfg["device"] = args.device
    if args.is_mi_based_edge is not None:
        cfg.setdefault("edge", {})["is_mi_based"] = bool(args.is_mi_based_edge)
    if args.edge_cache_path is not None:
        cfg.setdefault("edge", {})["cache_path"] = str(args.edge_cache_path)
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg.setdefault("train", {})["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = args.seed
    if args.binary is not None:
        cfg.setdefault("train", {})["binary"] = bool(args.binary)
    if args.decision_threshold is not None:
        cfg.setdefault("train", {})["decision_threshold"] = args.decision_threshold
    if args.cv is not None:
        cfg.setdefault("cv", {})["cv"] = args.cv
    return cfg

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = override_cfg(cfg, args)

    '''mi_cache_path = request_mi(
        mode="single",
        seed=cfg.get("seed", 1),
        fold=None,
        cfg=cfg,
        n_neighbors=cfg["edge"].get("n_neighbors", 1),
    )'''
    
    if cfg["train"]["cv"]:
        run_kfold_experiment(cfg, root)
    else:
        # run_single_experiment(cfg, root, mi_cache_path=mi_cache_path)
        run_single_experiment(cfg, root)

if __name__ == "__main__":
    main()