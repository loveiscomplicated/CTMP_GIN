import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root.parent))

import yaml
import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data_processing.tackle_missing_value import tackle_missing_value_wrapper
from src.data_processing.tensor_dataset import TEDSTensorDataset, TEDSDatasetForGIN
from src.data_processing.data_utils import train_test_split_stratified
from src.models.factory import build_model, build_edge
from src.trainers.base import run_train_loop
from src.utils.experiment import make_run_id, ensure_run_dir, ExperimentLogger
from src.utils.seed_set import set_seed
from src.utils.device_set import device_set
from src.trainers.utils.early_stopper import EarlyStopper

cur_dir = os.path.dirname(__file__)
root = os.path.join(cur_dir, 'data')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True) # config file location
    # overrides
    # p.add_argument("--model", type=str, default=None) no need, model selection only based on config
    # more detailed adjustment able in config file
    p.add_argument("--is_mi_based_edge", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--decision_threshold", type=float, default=None)
    # TODO non-binary (Multiclass Classification) implement later
    p.add_argument("--binary", type=int, default=None)
    return p.parse_args()

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def override_cfg(cfg: dict, args) -> dict:
    if args.device is not None:
        cfg["device"] = args.device
    if args.is_mi_based_edge is not None:
        cfg.setdefault("edge", {})["is_mi_based"] = bool(args.is_mi_based_edge)
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
        
    return cfg

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = override_cfg(cfg, args)

    run_id = make_run_id(cfg)
    run_dir = ensure_run_dir("runs", run_id)
    logger = ExperimentLogger(cfg, run_dir)

    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    device = device_set(cfg["device"])

    # create dataset
    dataset = TEDSTensorDataset(
        root=root,
        binary=cfg["train"].get("binary", True),
    )

    cfg["model"]["params"]["col_info"] = dataset.col_info
    cfg["model"]["params"]["num_classes"] = dataset.num_classes
    
    num_nodes = len(dataset.col_info[2]) # col_info: (col_list, col_dims, ad_col_index, dis_col_index)

    # create dataloaders
    split_ratio = [cfg['train']['train_ratio'], cfg['train']['val_ratio'], cfg['train']['test_ratio']]
    train_loader, val_loader, test_loader, idx = train_test_split_stratified(dataset=dataset,  # type: ignore
                                                                                   batch_size=cfg['train']['batch_size'],
                                                                                   ratio=split_ratio,
                                                                                   seed=seed,
                                                                                   num_workers=cfg['train']['num_workers'],
                                                                                   )
    
    train_df = dataset.processed_df.iloc[idx[0]]

    if cfg["model"]["name"] == "xgboost":
        from src.models.xgboost import train_xgboost
        train_idx, val_idx, test_idx = idx
        return train_xgboost(train_idx, val_idx, test_idx, dataset.processed_df, logger, cfg)

    # build model
    model = build_model(
        model_name=cfg["model"]["name"],
        **cfg["model"].get("params", {})
    )
    model = model.to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"학습 가능한 파라미터 개수: {total_trainable_params:,}")

    # build edge_index
    edge_index = build_edge(model_name=cfg["model"]["name"],
                            root=root,
                            seed=seed,
                            train_df=train_df,
                            num_nodes=num_nodes,
                            batch_size = cfg["train"]["batch_size"],
                            **cfg.get("edge", {})
                            )
    edge_index = edge_index.to(device) # type: ignore

    print(f'edge index: \n{edge_index}')
    print(f'edge index shape: \n{edge_index.shape}')

    if cfg["train"]["binary"]:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=cfg["train"]["lr_scheduler_patience"])
    early_stopper = EarlyStopper(patience=cfg["train"]["early_stopping_patience"])

    run_train_loop(
        model=model,
        edge_index=edge_index,
        binary=cfg["train"]["binary"],
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopper=early_stopper,
        device=device,
        logger=logger,
        epochs=cfg["train"]["epochs"],
        decision_threshold=cfg["train"]["decision_threshold"],
    )
    

if __name__ == "__main__":
    main()