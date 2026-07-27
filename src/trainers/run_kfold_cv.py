from __future__ import annotations

import copy
import json
import os
import shutil
import traceback
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_processing.splits import (
    holdout_test_split_stratified,
    kfold_stratified,
    make_loaders,
)
from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.models.factory import build_edge, build_model
from src.trainers.base import run_train_loop
from src.trainers.utils.early_stopper import EarlyStopper
from src.utils.device_set import device_set
from src.utils.experiment import (
    ExperimentLogger,
    _get_command_line,
    _get_git_info,
    make_run_id,
    save_text,
    save_yaml,
)
from src.utils.seed_set import set_seed


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=False)


def _cv_run_id(cfg: dict[str, Any]) -> str:
    k = cfg["train"]["n_folds"]
    test_ratio = cfg["train"]["test_ratio"]
    return make_run_id(cfg) + f"__cv={k}__test={test_ratio}"


def _cv_dir_from_cfg(cfg: dict[str, Any], cv_run_dir: str | None = None) -> str:
    if cv_run_dir is not None:
        return cv_run_dir
    return os.path.join("runs", _cv_run_id(cfg))


def _splits_path(cv_dir: str) -> str:
    return os.path.join(cv_dir, "kfold_splits.json")


def _fold_dir(cv_dir: str, fold: int) -> str:
    return os.path.join(cv_dir, "folds", f"fold_{fold}")


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_dataset(cfg: dict[str, Any], root: str) -> TEDSTensorDataset:
    admission_only = bool(cfg.get("admission_only", False))
    remove_los = True
    if not admission_only and cfg["model"]["name"] in [
        "gin",
        "a3tgcn_2_points",
        "gin_gru_2_points",
    ]:
        remove_los = False
        cfg.setdefault("edge", {})["remove_los"] = False

    return TEDSTensorDataset(
        root=root,
        binary=cfg["train"].get("binary", True),
        ig_label=cfg["train"].get("ig_label", False),
        remove_los=remove_los,
        do_preprocess=cfg["train"].get("do_preprocess", True),
        admission_only=admission_only,
    )


def _labels_from_dataset(dataset: TEDSTensorDataset) -> np.ndarray:
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))], dtype=np.int64)


def _set_model_params(
    cfg: dict[str, Any], dataset: TEDSTensorDataset, device: torch.device
) -> int:
    cfg["model"]["params"]["col_info"] = dataset.col_info
    cfg["model"]["params"]["num_classes"] = dataset.num_classes
    cfg["model"]["params"]["device"] = str(device)

    if bool(cfg.get("admission_only", False)):
        cfg["model"]["params"]["use_los"] = False

    if cfg["model"]["name"] in ["gin", "mlp"]:
        num_nodes = len(dataset.col_info[0])
    else:
        num_nodes = len(dataset.col_info[2])
    print(f"num_nodes set to {num_nodes}")
    return num_nodes


def _prepare_fold_dirs(cv_dir: str, fold: int) -> str:
    fold_dir = _fold_dir(cv_dir, fold)
    if os.path.exists(fold_dir):
        backup_dir = fold_dir + ".incomplete"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.move(fold_dir, backup_dir)
    os.makedirs(fold_dir, exist_ok=False)
    os.makedirs(os.path.join(fold_dir, "checkpoints"), exist_ok=True)
    return fold_dir


def _write_fold_status(cv_dir: str, fold: int, payload: dict[str, Any]) -> None:
    _write_json(os.path.join(_fold_dir(cv_dir, fold), "fold_status.json"), payload)


def _save_fold_result(cv_dir: str, fold: int, result: dict[str, Any]) -> None:
    _write_json(os.path.join(_fold_dir(cv_dir, fold), "fold_result.json"), result)


def _load_fold_splits(cv_dir: str) -> dict[str, Any]:
    return _load_json(_splits_path(cv_dir))


def prepare_kfold_run(
    cfg: dict[str, Any], root: str, cv_run_dir: str | None = None
) -> dict[str, Any]:
    cv_dir = _cv_dir_from_cfg(cfg, cv_run_dir)
    ensure_dir(cv_dir)
    ensure_dir(os.path.join(cv_dir, "folds"))

    save_yaml(os.path.join(cv_dir, "config.final.yaml"), cfg)
    save_text(os.path.join(cv_dir, "command.txt"), _get_command_line() + "\n")
    save_text(os.path.join(cv_dir, "git.txt"), _get_git_info())

    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    dataset = _build_dataset(copy.deepcopy(cfg), root)
    labels = _labels_from_dataset(dataset)
    trainval_idx, test_idx = holdout_test_split_stratified(
        dataset=dataset,
        test_ratio=cfg["train"]["test_ratio"],
        seed=seed,
        labels=labels,
    )

    folds: list[dict[str, Any]] = []
    for fold, train_idx, val_idx in kfold_stratified(
        trainval_idx=trainval_idx,
        labels=labels,
        n_folds=cfg["train"]["n_folds"],
        seed=seed,
    ):
        folds.append(
            {
                "fold": int(fold),
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
            }
        )

    splits = {
        "cv_id": os.path.basename(cv_dir),
        "split_mode": "holdout_test_plus_kfold_val",
        "seed": int(seed),
        "test_ratio": float(cfg["train"]["test_ratio"]),
        "n_folds": int(cfg["train"]["n_folds"]),
        "trainval_idx": trainval_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "folds": folds,
    }
    _write_json(_splits_path(cv_dir), splits)
    return {"cv_dir": cv_dir, "splits_path": _splits_path(cv_dir), "splits": splits}


def _find_fold_split(splits: dict[str, Any], fold: int) -> dict[str, Any]:
    for fold_info in splits["folds"]:
        if int(fold_info["fold"]) == fold:
            return fold_info
    raise ValueError(f"Fold {fold} not found in saved splits")


def run_single_fold(
    cfg: dict[str, Any],
    root: str,
    fold: int,
    cv_run_dir: str,
    *,
    resume_from_last: bool = False,
) -> dict[str, Any]:
    if resume_from_last:
        raise NotImplementedError(
            "--resume_fold_from_last is not supported in the split CTMP-GIN runner."
        )

    splits = _load_fold_splits(cv_run_dir)
    fold_split = _find_fold_split(splits, fold)
    fold_dir = _prepare_fold_dirs(cv_run_dir, fold)
    _write_fold_status(
        cv_run_dir,
        fold,
        {"fold": fold, "status": "running", "run_dir": fold_dir},
    )

    try:
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["fold"] = fold

        seed = fold_cfg["train"].get("seed", 42)
        set_seed(seed)
        device = device_set(fold_cfg["device"])

        dataset = _build_dataset(fold_cfg, root)
        num_nodes = _set_model_params(fold_cfg, dataset, device)
        fold_logger = ExperimentLogger(fold_cfg, fold_dir)

        train_idx = np.asarray(fold_split["train_idx"], dtype=np.int64)
        val_idx = np.asarray(fold_split["val_idx"], dtype=np.int64)
        test_idx = np.asarray(splits["test_idx"], dtype=np.int64)
        requires_fixed_batch_edge = fold_cfg["model"]["name"] in ["gin_gru"]
        drop_last = bool(fold_cfg["train"].get("drop_last", requires_fixed_batch_edge))
        train_loader, val_loader, test_loader = make_loaders(
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=fold_cfg["train"]["batch_size"],
            num_workers=fold_cfg["train"]["num_workers"],
            drop_last=drop_last,
            pin_memory=bool(fold_cfg["train"].get("pin_memory", device.type == "cuda")),
            persistent_workers=fold_cfg["train"].get("persistent_workers", None),
            prefetch_factor=fold_cfg["train"].get("prefetch_factor", None),
        )

        if fold_cfg["model"]["name"] == "xgboost":
            from src.models.xgboost import train_xgboost

            result = train_xgboost(
                train_idx,
                val_idx,
                test_idx,
                dataset.processed_df,
                fold_logger,
                fold_cfg,
            )
            result["fold"] = fold
            result["run_dir"] = fold_dir
            result["status"] = "completed"
            _save_fold_result(cv_run_dir, fold, result)
            _write_fold_status(
                cv_run_dir,
                fold,
                {"fold": fold, "status": "completed", "run_dir": fold_dir},
            )
            return result

        if fold_cfg["model"]["name"] in ["a3tgcn", "a3tgcn_2_points"]:
            fold_cfg["model"]["params"]["batch_size"] = fold_cfg["train"].get(
                "batch_size", 32
            )

        model = build_model(
            model_name=fold_cfg["model"]["name"],
            **fold_cfg["model"].get("params", {}),
        ).to(device)
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(model)
        print(f"학습 가능한 파라미터 개수: {total_trainable_params:,}")

        train_df = dataset.processed_df.iloc[train_idx]
        edge_index = build_edge(
            model_name=fold_cfg["model"]["name"],
            root=root,
            seed=seed,
            train_df=train_df,
            num_nodes=num_nodes,
            batch_size=fold_cfg["train"]["batch_size"],
            **fold_cfg.get("edge", {}),
        ).to(device)

        if hasattr(model, "precompute_edge_index_2"):
            model.precompute_edge_index_2(edge_index, fold_cfg["train"]["batch_size"])

        if bool(fold_cfg["train"].get("compile", False)) and device.type == "cuda":
            compile_mode = fold_cfg["train"].get("compile_mode", "default")
            model = torch.compile(model, mode=compile_mode)

        edge_index_save_path = os.path.join(fold_dir, "edge_index.pt")
        torch.save(edge_index.cpu(), edge_index_save_path)
        print(f"edge_index saved: {edge_index_save_path}")
        print(f"edge index shape: {tuple(edge_index.shape)}")
        print(f"edge count: {edge_index.size(1):,}")

        if fold_cfg["train"]["binary"]:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer_name = fold_cfg["train"].get("optimizer", "adam")
        optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
        optimizer_kwargs = {
            "lr": fold_cfg["train"]["learning_rate"],
            "weight_decay": fold_cfg["train"].get("weight_decay", 0.0),
        }
        if device.type == "cuda" and bool(fold_cfg["train"].get("fused_optimizer", True)):
            optimizer_kwargs["fused"] = True
        try:
            optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        except (TypeError, RuntimeError):
            optimizer_kwargs.pop("fused", None)
            optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            patience=fold_cfg["train"]["lr_scheduler_patience"],
        )
        early_stopper = EarlyStopper(
            patience=fold_cfg["train"]["early_stopping_patience"]
        )
        results = run_train_loop(
            model=model,
            edge_index=edge_index,
            binary=fold_cfg["train"]["binary"],
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=early_stopper,
            device=device,
            logger=fold_logger,
            epochs=fold_cfg["train"]["epochs"],
            decision_threshold=fold_cfg["train"]["decision_threshold"],
            model_name=fold_cfg["model"].get("name", "Unknown"),
            trial=None,
            amp=fold_cfg["train"].get("amp", device.type == "cuda"),
            tf32=fold_cfg["train"].get("tf32", device.type == "cuda"),
            disable_tqdm=fold_cfg["train"].get("disable_tqdm", False),
        )

        results["fold"] = fold
        results["run_dir"] = fold_dir
        results["status"] = "completed"
        _save_fold_result(cv_run_dir, fold, results)
        _write_fold_status(
            cv_run_dir,
            fold,
            {"fold": fold, "status": "completed", "run_dir": fold_dir},
        )
        return results

    except Exception as exc:
        error_payload = {
            "fold": fold,
            "status": "failed",
            "run_dir": fold_dir,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_fold_status(cv_run_dir, fold, error_payload)
        raise


def _aggregate_numeric_metrics(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    keys: set[str] = set()
    for result in results:
        for key, value in result.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and key != "fold":
                keys.add(key)

    summary: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        values = [
            float(result[key])
            for result in results
            if key in result and np.isfinite(float(result[key]))
        ]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": float(len(values)),
            }
    return summary


def finalize_kfold_summary(cv_run_dir: str) -> dict[str, Any]:
    splits = _load_fold_splits(cv_run_dir)
    results: list[dict[str, Any]] = []
    missing: list[int] = []
    failed: list[int] = []

    for fold_info in splits["folds"]:
        fold = int(fold_info["fold"])
        result_path = os.path.join(_fold_dir(cv_run_dir, fold), "fold_result.json")
        status_path = os.path.join(_fold_dir(cv_run_dir, fold), "fold_status.json")
        if os.path.exists(result_path):
            result = _load_json(result_path)
            results.append(result)
            continue
        if os.path.exists(status_path):
            status = _load_json(status_path)
            if status.get("status") == "failed":
                failed.append(fold)
            else:
                missing.append(fold)
        else:
            missing.append(fold)

    summary = {
        "status": "completed" if len(results) == len(splits["folds"]) else "incomplete",
        "cv_run_dir": cv_run_dir,
        "completed_folds": len(results),
        "expected_folds": len(splits["folds"]),
        "missing_folds": missing,
        "failed_folds": failed,
        "metrics": _aggregate_numeric_metrics(results),
        "fold_results": results,
    }
    _write_json(os.path.join(cv_run_dir, "kfold_summary.json"), summary)
    return summary


def run_kfold_experiment(cfg: dict[str, Any], root: str) -> dict[str, Any]:
    prepared = prepare_kfold_run(cfg, root)
    cv_dir = prepared["cv_dir"]
    for fold_info in prepared["splits"]["folds"]:
        run_single_fold(cfg, root, int(fold_info["fold"]), cv_dir)
    return finalize_kfold_summary(cv_dir)
