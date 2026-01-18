import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root.parent))

import yaml
import argparse
import pandas as pd
from src.data_processing.tackle_missing_value import tackle_missing_value_main
from src.data_processing.tensor_dataset import TEDSTensorDataset, TEDSDatasetForGIN
from src.data_processing.data_utils import train_test_split_stratified
from src.data_processing.process_mi_dict import search_mi_dict
from src.data_processing.edge import mi_edge_index_batched, mi_edge_index_batched_for_baseline
from src.models.factory import build_model
from src.utils.experiment import make_run_id, ensure_run_dir, ExperimentLogger
from src.utils.seed_set import set_seed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True) # config file location
    # overrides
    # p.add_argument("--model", type=str, default=None) no need, model selection only based on config
    # more detailed adadjustment able in config file
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    # TODO non-binary (Multiclass Classification) implement later
    # p.add_argument("--binary", type=int, default=None)
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


    CUR_DIR = os.path.dirname(__file__)
    raw_df_path = os.path.join(CUR_DIR, 'data', 'raw', 'TEDS_Discharge.csv')

    # missing_corrected.csv will be created after `tackle_missing_value_main`.
    # missing_corrected.csv is used in every training processes.
    missing_corrected_path = os.path.join(CUR_DIR, 'data', 'raw', 'missing_corrected.csv')
    root = os.path.join(CUR_DIR, 'data')

    missing_corrected = tackle_missing_value_main(raw_df_path, missing_corrected_path)

    # TODO make dataset class


if __name__ == "__main__":
    main()