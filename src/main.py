import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root.parent))

import yaml
import argparse
import pandas as pd
from src.data_processing.tackle_missing_value import tackle_missing_value_wrapper
from src.data_processing.tensor_dataset import TEDSTensorDataset, TEDSDatasetForGIN
from src.data_processing.data_utils import train_test_split_stratified
from src.models.factory import build_model, build_edge
from src.trainers.base import train, evaluate
from src.utils.experiment import make_run_id, ensure_run_dir, ExperimentLogger
from src.utils.seed_set import set_seed
from src.utils.device_set import device_set

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
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
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
    if args.lr is not None:
        cfg.setdefault("train", {})["lr"] = args.lr
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = args.seed
    if args.binary is not None:
        cfg.setdefault("train", {})["binary"] = bool(args.binary)
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
    num_nodes = len(dataset.col_info[2]) # col_info: (col_list, col_dims, ad_col_index, dis_col_index)

    # create dataloaders
    split_ratio = [cfg['train']['train_ratio'], cfg['train']['val_ratio'], cfg['train']['test_ratio']]
    train_loader, val_loader, test_loader, train_idx = train_test_split_stratified(dataset=dataset, 
                                                                                   batch_size=cfg['train']['batch_size'],
                                                                                   ratio=split_ratio,
                                                                                   seed=cfg['train']['seed'],
                                                                                   num_workers=cfg['train']['num_workers'])
    train_df = dataset.processed_df.iloc[train_idx]

    # build model
    model = build_model(
        model_name=cfg["model"]["name"],
        device=device,
        **cfg["model"].get("params", {})
    )
    model.to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"학습 가능한 파라미터 개수: {total_trainable_params:,}")

    # build edge_index
    edge_index = build_edge(model_name=cfg["model"]["name"],
                            root=root,
                            seed=seed,
                            train_df=train_df,
                            device=device,
                            num_nodes=num_nodes,
                            batch_size = cfg["train"]["batch_size"],
                            **cfg.get("edge", {})
                            )
    edge_index.to(device) # type: ignore

    print(f'edge index: \n{edge_index}')
    print(f'edge index shape: \n{edge_index.shape}')


    # TODO train loop 생성
    '''train(
        model=model,
        dataloader=train_loader,
        device=device,
        edge_index=edge_index,
        **cfg.get("train", {})
    )'''


    

if __name__ == "__main__":
    main()