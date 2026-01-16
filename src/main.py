import os
import yaml
import argparse
import pandas as pd
from data_processing.tensor_dataset import TEDSTensorDataset, TEDSDatasetForGIN
from data_processing.data_utils import train_test_split_stratified
from data_processing.process_mi_dict import search_mi_dict
from data_processing.edge import mi_edge_index_batched, mi_edge_index_batched_for_baseline
from models.factory import build_model
from utils.experiment import make_run_id, ensure_run_dir, ExperimentLogger
from utils.seed_set import set_seed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    # overrides
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--binary", type=int, default=None)
    return p.parse_args()

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def override_cfg(cfg: dict, args) -> dict:
    if args.model is not None:
        cfg.setdefault("model", {})["name"] = args.model

    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg.setdefault("train", {})["lr"] = args.lr
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = args.seed

    if args.binary is not None:
        cfg.setdefault("data", {})["binary"] = bool(args.binary)

    return cfg

def main():
    # 1~3
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = override_cfg(cfg, args)

    # ------------------------
    # 5) run dir + logger
    # ------------------------
    run_id = make_run_id(cfg)
    run_dir = ensure_run_dir("runs", run_id)
    logger = ExperimentLogger(cfg, run_dir)

    # ------------------------
    # 6) seed
    # ------------------------
    seed = cfg["train"].get("seed", 42)
    set_seed(seed)  # 네가 쓰는 seed 함수

    # ------------------------
    # 7) dataset
    # ------------------------
    dataset = TEDSTensorDataset(
        root=cfg["data"]["root"],
        binary=cfg["data"].get("binary", True),
    )

    # ------------------------
    # 8) model
    # ------------------------
    model = build_model(
        cfg["model"]["name"],
        **cfg["model"].get("params", {})
    )

    # ------------------------
    # 9) train
    # ------------------------
    train(
        model=model,
        dataset=dataset,
        cfg=cfg,
        logger=logger,
    )




cur_dir = os.path.dirname(__file__)
run_dir = os.path.join(cur_dir, 'runs')
root = os.path.join(cur_dir, 'data')


run_id = make_run_id(cfg)
run_dir = ensure_run_dir("runs", run_id)
logger = ExperimentLogger(cfg, run_dir)


