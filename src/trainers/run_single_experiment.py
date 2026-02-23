import os
import optuna
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data_processing.tensor_dataset import TEDSTensorDataset, TEDSDatasetForGIN
from src.data_processing.data_utils import train_test_split_stratified
from src.models.factory import build_model, build_edge
from src.trainers.base import run_train_loop
from src.utils.experiment import make_run_id, ensure_run_dir, ExperimentLogger
from src.utils.seed_set import set_seed
from src.utils.device_set import device_set
from src.trainers.utils.early_stopper import EarlyStopper

def run_single_experiment(cfg, 
                          root,
                          **kwargs):
    report_metric = kwargs.get("report_metric", "valid_auc")
    trial = kwargs.get("trial", None)
    edge_cached=kwargs["edge"].get("edge_cached", True)

    logger = None
    if trial is None: # if not parameter searching (normal training session)
        run_id = make_run_id(cfg)
        run_dir = ensure_run_dir("runs", run_id)
        logger = ExperimentLogger(cfg, run_dir) # if parameter searching, turn off the logger

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
    cfg["model"]["params"]["device"] = device
    
    num_nodes = len(dataset.col_info[2]) # col_info: (col_list, col_dims, ad_col_index, dis_col_index)

    if cfg["model"]["name"] == 'gin':
        num_nodes = len(dataset.col_info[0]) + 1

    print(f"num_nodes set to {num_nodes}")

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
    
    if cfg["model"]["name"] == "a3tgcn":
        cfg["model"]["params"]["batch_size"] = cfg["train"].get("batch_size", 32)

    # build model
    model = build_model(
        model_name=cfg["model"]["name"],
        **cfg["model"].get("params", {})
    )
    model = model.to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


    # build edge_index
    edge_index = build_edge(model_name=cfg["model"]["name"],
                            root=root,
                            seed=seed,
                            train_df=train_df,
                            num_nodes=num_nodes,
                            batch_size = cfg["train"]["batch_size"],
                            edge_cached=edge_cached,
                            **cfg.get("edge", {})
                            )
    edge_index = edge_index.to(device) # type: ignore
    if trial is None:
        print(model)
        print(f"학습 가능한 파라미터 개수: {total_trainable_params:,}")
        print(f'edge index: \n{edge_index}')
        print(f'edge index shape: \n{edge_index.shape}')

    if cfg["train"]["binary"]:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    if cfg["train"].get("optimizer", "adam") == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=cfg["train"]["learning_rate"], 
                                    weight_decay=cfg["train"].get("weight_decay", 0.0))
        
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                      lr=cfg["train"]["learning_rate"], 
                                      weight_decay=cfg["train"].get("weight_decay", 0.0))

    scheduler = ReduceLROnPlateau(optimizer, "min", patience=cfg["train"]["lr_scheduler_patience"])
    early_stopper = EarlyStopper(patience=cfg["train"]["early_stopping_patience"])

    out = run_train_loop(
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
        trial=trial,
        report_metric=report_metric,
    )

    return out