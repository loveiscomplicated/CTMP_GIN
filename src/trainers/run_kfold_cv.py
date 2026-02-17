import os
import json
import numpy as np
import copy
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.seed_set import set_seed
from src.data_processing.splits import holdout_test_split_stratified, kfold_stratified, make_loaders
from src.data_processing.tensor_dataset import TEDSTensorDataset, TEDSDatasetForGIN
from src.models.factory import build_model, build_edge
from src.trainers.base import run_train_loop
from src.utils.experiment import (make_run_id, 
                                  save_text, 
                                  save_yaml, 
                                  _get_command_line, 
                                  _get_git_info,
                                  ExperimentLogger)
from src.utils.device_set import device_set
from src.trainers.utils.early_stopper import EarlyStopper


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=False)

def run_kfold_experiment(cfg, root):
    K = cfg["train"]["n_folds"]
    test_ratio = cfg["train"]["test_ratio"]

    # 1) CV 상위 폴더
    cv_id = make_run_id(cfg) + f"__cv={K}__test={test_ratio}"
    cv_dir = os.path.join("runs", cv_id)
    ensure_dir(cv_dir)
    ensure_dir(os.path.join(cv_dir, "folds"))

    # (선택) CV 상위에도 config/command/git 저장하고 싶으면:
    save_yaml(os.path.join(cv_dir, "config.final.yaml"), cfg)
    save_text(os.path.join(cv_dir, "command.txt"), _get_command_line() + "\n")
    save_text(os.path.join(cv_dir, "git.txt"), _get_git_info())

    seed = cfg["train"].get("seed", 42)
    set_seed(seed) 

    device = device_set(cfg["device"])

    # 2) dataset 생성 + split 생성 (indices)
    # create dataset
    if cfg["model"]["name"] == 'gin':
        dataset = TEDSTensorDataset(
            root=root,
            binary=cfg["train"].get("binary", True),
            ig_label=False, # we don't train models for IG in k-fold cv
            remove_los=False # only diff is maintatin los in processed_df --> related to getting mi_dict
        )
    else:
        dataset = TEDSTensorDataset(
            root=root,
            binary=cfg["train"].get("binary", True),
            ig_label=False, # we don't train models for IG in k-fold cv
            remove_los=True
        )

    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    cfg["model"]["params"]["col_info"] = dataset.col_info
    cfg["model"]["params"]["num_classes"] = dataset.num_classes
    cfg["model"]["params"]["device"] = str(device)
    num_nodes = len(dataset.col_info[2]) # col_info: (col_list, col_dims, ad_col_index, dis_col_index)

    if cfg["model"]["name"] == 'gin':
        num_nodes = len(dataset.col_info[0]) + 1 # variables + LOS (except REASON)

    trainval_idx, test_idx = holdout_test_split_stratified(
        dataset=dataset,
        test_ratio=test_ratio,
        seed=seed,
        labels=labels
    )

    fold_results = []

    for fold, train_idx, val_idx in kfold_stratified(
        trainval_idx=trainval_idx,
        labels=labels,
        n_folds=K,
        seed=seed
    ):
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["fold"] = fold

        fold_dir = os.path.join(cv_dir, "folds", f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=False)
        os.makedirs(os.path.join(fold_dir, "checkpoints"), exist_ok=True)

        fold_logger = ExperimentLogger(fold_cfg, fold_dir)

        # fold별로 loaders / train_df / edge_index / model 새로
        # run_train_loop(..., logger=fold_logger)
        # result dict에 fold metric 담아서 append

        train_loader, val_loader, test_loader = make_loaders(
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=cfg['train']['batch_size'],
            num_workers=cfg['train']['num_workers'],
            drop_last=True
        )

        print(len(train_loader))

        train_df = dataset.processed_df.iloc[train_idx]

        if cfg["model"]["name"] == "xgboost":
            from src.models.xgboost import train_xgboost
            result = train_xgboost(train_idx, val_idx, test_idx, dataset.processed_df, fold_logger, cfg)
            fold_results.append(result)
            continue
        
        if cfg["model"]["name"] == "a3tgcn":
            cfg["model"]["params"]["batch_size"] = cfg["train"].get("batch_size", 32)

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
                                edge_cached=False,
                                **cfg.get("edge", {})
                                )
        edge_index = edge_index.to(device) # type: ignore

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
            
        else: # if adam
            optimizer = torch.optim.Adam(model.parameters(),
                                        lr=cfg["train"]["learning_rate"], 
                                        weight_decay=cfg["train"].get("weight_decay", 0.0))
            
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
            logger=fold_logger,
            epochs=cfg["train"]["epochs"],
            decision_threshold=cfg["train"]["decision_threshold"],
        )

        fold_results.append({"fold": fold, "run_dir": fold_dir})

    # 3) CV 요약 저장
    summary = {
        "cv_id": cv_id,
        "K": K,
        "test_ratio": test_ratio,
        "fold_results": fold_results,
        # 여기에 mean/std 추가
    }
    with open(os.path.join(cv_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
