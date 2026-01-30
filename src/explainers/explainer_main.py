import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root.parent))

import yaml
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data_processing.tackle_missing_value import tackle_missing_value_wrapper
from src.data_processing.tensor_dataset import TEDSTensorDataset, TEDSDatasetForGIN
from src.data_processing.data_utils import train_test_split_stratified
from src.models.factory import build_model, build_edge
from src.trainers.base import load_checkpoint
from src.utils.experiment import make_run_id, ensure_run_dir, ExperimentLogger
from src.utils.seed_set import set_seed
from src.utils.device_set import device_set
from src.trainers.utils.early_stopper import EarlyStopper

from src.explainers.gb_ig import CTMPGIN_GBIGExplainer, compute_global_importance_on_loader, GlobalImportanceOutput
from src.explainers.stablity_report import (
    stability_report, 
    print_stability_report, 
    topk_indices, 
    unstable_variables_report, 
    print_unstable_report_with_names,
    importance_mean_std_table
)

cur_dir = os.path.dirname(__file__)
root = os.path.join(cur_dir, '..', 'data')
save_path = os.path.join(cur_dir, 'results')
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

# --------@@@@ adjust model path !!! @@@@--------
model_path = os.path.join(cur_dir, '..', '..', 'runs', 'temp_ctmp_gin_ckpt', '1218_ctmp_epoch_37_loss_0.2717.pth')

# TODO add mode arg to select the explaination method, default: None -> do all method, make save path align according to this mode, add detailed configurations in this path 
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

    # run_id = make_run_id(cfg)
    # run_dir = ensure_run_dir("runs", run_id)
    # logger = ExperimentLogger(cfg, run_dir)

    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    device = device_set(cfg["device"])

    # create dataset
    dataset = TEDSTensorDataset(
        root=root,
        binary=cfg["train"].get("binary", True),
        ig_label=cfg["train"].get("ig_label", False),
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

    '''    
    if cfg["model"]["name"] == "xgboost":
        from src.models.xgboost import train_xgboost
        train_idx, val_idx, test_idx = idx
        return train_xgboost(train_idx, val_idx, test_idx, dataset.processed_df, logger, cfg)
    
    if cfg["model"]["name"] == "a3tgcn":
        cfg["model"]["params"]["batch_size"] = cfg["train"].get("batch_size", 32)
        cfg["model"]["params"]["device"] = device
    '''

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

    load_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        filename=model_path
    )    
    
    explainer = CTMPGIN_GBIGExplainer(
        model=model,
        edge_index_vargraph=edge_index.detach().cpu(),
        ad_indices=dataset.col_info[2],  # type: ignore
        dis_indices=dataset.col_info[3], # type: ignore
        baseline_strategy="farthest",
        max_paths=1,            
        use_abs=True,
        device=device,
    )
    
    rat = 0.05
    outs = []
    for s in [0, 1, 2]:
        set_seed(s)
        out = compute_global_importance_on_loader(
            explainer=explainer,
            model=model,
            dataloader=test_loader,
            edge_index=edge_index,
            device=device,
            sample_ratio=rat,
            seed=s,
            keep_all=False,
            reduce="mean",
            verbose=True,
        )
        outs.append(out.global_importance.cpu().float())  # [N]
    
    col_names = dataset.col_info[0]

    # ---- after outs computed ----
    df_ms = importance_mean_std_table(outs, col_names)

    # 보기 편하게 top/bottom
    print("\n=== Top 20 (mean ± std) ===")
    print(df_ms.head(20).to_string(index=False))

    print("\n=== Bottom 20 (mean ± std) ===")
    print(df_ms.tail(20).to_string(index=False))

    # CSV 저장
    out_csv = os.path.join(save_path, "gbig_global_importance_mean_std.csv")
    df_ms.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # get reports
    report = stability_report(outs, ks=[10, 20, 30])
    print_stability_report(report, ks=[10, 20, 30])

    rep20 = unstable_variables_report(outs, k=20)
    print_unstable_report_with_names(rep20, col_names)

    rep30 = unstable_variables_report(outs, k=30)
    print_unstable_report_with_names(rep30, col_names)
    
    # TODO include los into variable importances

if __name__ == "__main__":
    main()