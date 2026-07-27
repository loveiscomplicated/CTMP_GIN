from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import yaml

from src.data_processing.data_utils import train_test_split_stratified
from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.models.factory import build_edge, build_model
from src.trainers import base as trainer_base


TARGET_MODELS = {"ctmp_gin", "gin", "gin_gru_2_points", "a3tgcn_2_points"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile real-data training without writing checkpoints."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="src/data")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--phase-steps", type=int, default=5)
    parser.add_argument("--throughput-steps", type=int, default=20)
    parser.add_argument("--throughput-repeats", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def device_memory_mib(device: torch.device) -> dict[str, float]:
    if device.type == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(device) / 2**20,
            "reserved": torch.cuda.memory_reserved(device) / 2**20,
            "peak_allocated": torch.cuda.max_memory_allocated(device) / 2**20,
        }
    if device.type == "mps":
        return {
            "allocated": torch.mps.current_allocated_memory() / 2**20,
            "driver": torch.mps.driver_allocated_memory() / 2**20,
            "recommended_max": torch.mps.recommended_max_memory() / 2**20,
        }
    return {}


def median_ms(values: list[float]) -> float:
    return 1000.0 * statistics.median(values) if values else 0.0


class CyclingIterator:
    def __init__(self, loader) -> None:
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def move_batch(batch, device: torch.device):
    x_batch, y_batch, los_batch = batch
    non_blocking = device.type == "cuda"
    return (
        x_batch.to(device, non_blocking=non_blocking),
        y_batch.to(device, non_blocking=non_blocking),
        los_batch.to(device, non_blocking=non_blocking),
    )


def compute_loss(
    model: nn.Module,
    batch,
    edge_index: torch.Tensor,
    criterion: nn.Module,
    binary: bool,
    device: torch.device,
) -> torch.Tensor:
    x_batch, y_batch, los_batch = batch
    logits = model(x_batch, los_batch, edge_index, device=device)
    if binary:
        return criterion(logits.squeeze(1), y_batch.float())
    return criterion(logits, y_batch.long())


def phase_profile(
    model: nn.Module,
    loader,
    edge_index: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    binary: bool,
    device: torch.device,
    steps: int,
) -> dict[str, float]:
    iterator = CyclingIterator(loader)
    samples: dict[str, list[float]] = {
        "data_wait": [],
        "h2d": [],
        "forward_loss": [],
        "backward": [],
        "optimizer": [],
    }

    model.train()
    for _ in range(steps):
        start = time.perf_counter()
        batch_cpu = iterator.next()
        samples["data_wait"].append(time.perf_counter() - start)

        start = time.perf_counter()
        batch = move_batch(batch_cpu, device)
        synchronize(device)
        samples["h2d"].append(time.perf_counter() - start)

        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        loss = compute_loss(model, batch, edge_index, criterion, binary, device)
        synchronize(device)
        samples["forward_loss"].append(time.perf_counter() - start)

        start = time.perf_counter()
        loss.backward()
        synchronize(device)
        samples["backward"].append(time.perf_counter() - start)

        start = time.perf_counter()
        optimizer.step()
        synchronize(device)
        samples["optimizer"].append(time.perf_counter() - start)

    return {f"{name}_median_ms": median_ms(values) for name, values in samples.items()}


def run_training_steps(
    model: nn.Module,
    iterator: CyclingIterator,
    edge_index: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    binary: bool,
    device: torch.device,
    steps: int,
) -> None:
    model.train()
    for _ in range(steps):
        batch = move_batch(iterator.next(), device)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, batch, edge_index, criterion, binary, device)
        loss.backward()
        optimizer.step()


def throughput_profile(
    model: nn.Module,
    loader,
    edge_index: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    binary: bool,
    device: torch.device,
    warmup_steps: int,
    steps: int,
    repeats: int,
) -> dict[str, float]:
    iterator = CyclingIterator(loader)
    run_training_steps(
        model,
        iterator,
        edge_index,
        criterion,
        optimizer,
        binary,
        device,
        warmup_steps,
    )
    synchronize(device)

    seconds_per_step: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_training_steps(
            model,
            iterator,
            edge_index,
            criterion,
            optimizer,
            binary,
            device,
            steps,
        )
        synchronize(device)
        seconds_per_step.append((time.perf_counter() - start) / steps)

    median_step = statistics.median(seconds_per_step)
    batch_size = int(loader.batch_size)
    return {
        "step_median_ms": median_step * 1000.0,
        "samples_per_second": batch_size / median_step,
        "projected_train_epoch_seconds": median_step * len(loader),
    }


def validation_throughput_profile(
    model: nn.Module,
    loader,
    edge_index: torch.Tensor,
    criterion: nn.Module,
    binary: bool,
    device: torch.device,
    steps: int,
) -> dict[str, float]:
    iterator = CyclingIterator(loader)
    model.eval()
    measured_steps = min(steps, len(loader))

    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(measured_steps):
            x_batch, y_batch, los_batch = move_batch(iterator.next(), device)
            logits = model(x_batch, los_batch, edge_index, device=device)
            if binary:
                logits = logits.squeeze(1)
                loss = criterion(logits, y_batch.float())
                scores = torch.sigmoid(logits)
                predicted = (scores >= 0.5).long()
                scores.detach().cpu().numpy()
            else:
                loss = criterion(logits, y_batch.long())
                scores = torch.softmax(logits, dim=1)
                predicted = torch.argmax(scores, dim=1)
                scores.detach().cpu().numpy()
            loss.item()
            y_batch.detach().cpu().numpy()
            predicted.detach().cpu().numpy()
    synchronize(device)

    seconds_per_step = (time.perf_counter() - start) / measured_steps
    return {
        "sampled_steps": measured_steps,
        "step_seconds": seconds_per_step,
        "samples_per_second": int(loader.batch_size) / seconds_per_step,
        "projected_seconds": seconds_per_step * len(loader),
    }


def timed_phase(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def build_profile(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    cfg = copy.deepcopy(cfg)

    model_name = str(cfg["model"]["name"])
    if model_name not in TARGET_MODELS:
        raise ValueError(
            f"Unsupported model {model_name!r}; expected one of {sorted(TARGET_MODELS)}"
        )

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this process")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this process")

    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    binary = bool(cfg["train"].get("binary", True))
    admission_only = bool(cfg.get("admission_only", False))
    remove_los = not (
        not admission_only
        and model_name in {"gin", "a3tgcn_2_points", "gin_gru_2_points"}
    )

    dataset, dataset_seconds = timed_phase(
        lambda: TEDSTensorDataset(
            root=args.data_root,
            binary=binary,
            ig_label=cfg["train"].get("ig_label", False),
            remove_los=remove_los,
            do_preprocess=cfg["train"].get("do_preprocess", True),
            admission_only=admission_only,
        )
    )

    cfg["model"]["params"]["col_info"] = dataset.col_info
    cfg["model"]["params"]["num_classes"] = dataset.num_classes
    cfg["model"]["params"]["device"] = str(device)

    if model_name in {"gin"}:
        num_nodes = len(dataset.col_info[0])
    else:
        num_nodes = len(dataset.col_info[2])

    split_ratio = [
        cfg["train"]["train_ratio"],
        cfg["train"]["val_ratio"],
        cfg["train"]["test_ratio"],
    ]
    loaders, split_seconds = timed_phase(
        lambda: train_test_split_stratified(
            dataset=dataset,
            batch_size=cfg["train"]["batch_size"],
            ratio=split_ratio,
            seed=seed,
            num_workers=args.num_workers,
        )
    )
    train_loader, val_loader, test_loader, indices = loaders
    train_idx, _, _ = indices

    train_df, train_df_seconds = timed_phase(
        lambda: dataset.processed_df.iloc[train_idx]
    )

    if model_name in {"a3tgcn_2_points"}:
        cfg["model"]["params"]["batch_size"] = cfg["train"]["batch_size"]

    model, model_build_seconds = timed_phase(
        lambda: build_model(model_name=model_name, **cfg["model"]["params"])
    )
    model = model.to(device)

    edge_index, edge_build_seconds = timed_phase(
        lambda: build_edge(
            model_name=model_name,
            root=args.data_root,
            seed=seed,
            train_df=train_df,
            num_nodes=num_nodes,
            batch_size=cfg["train"]["batch_size"],
            **cfg.get("edge", {}),
        )
    )
    if isinstance(edge_index, tuple):
        edge_index = edge_index[0]

    synchronize(device)
    start = time.perf_counter()
    edge_index = edge_index.to(device)
    if hasattr(model, "precompute_edge_index_2"):
        model.precompute_edge_index_2(edge_index, cfg["train"]["batch_size"])
    synchronize(device)
    edge_transfer_seconds = time.perf_counter() - start

    criterion: nn.Module
    if binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer_cls = (
        torch.optim.AdamW
        if cfg["train"].get("optimizer", "adam") == "adamw"
        else torch.optim.Adam
    )
    optimizer = optimizer_cls(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    start = time.perf_counter()
    first_iterator = iter(train_loader)
    first_batch = next(first_iterator)
    first_batch_seconds = time.perf_counter() - start
    del first_batch, first_iterator

    throughput = throughput_profile(
        model=model,
        loader=train_loader,
        edge_index=edge_index,
        criterion=criterion,
        optimizer=optimizer,
        binary=binary,
        device=device,
        warmup_steps=args.warmup_steps,
        steps=args.throughput_steps,
        repeats=args.throughput_repeats,
    )
    phases = phase_profile(
        model=model,
        loader=train_loader,
        edge_index=edge_index,
        criterion=criterion,
        optimizer=optimizer,
        binary=binary,
        device=device,
        steps=args.phase_steps,
    )

    validation: dict[str, Any] = {
        "batches": len(val_loader),
        **validation_throughput_profile(
            model=model,
            loader=val_loader,
            edge_index=edge_index,
            criterion=criterion,
            binary=binary,
            device=device,
            steps=args.eval_steps,
        ),
    }
    if args.full_validation:
        original_tqdm = trainer_base.tqdm
        trainer_base.tqdm = lambda iterable, **_: iterable
        try:
            synchronize(device)
            start = time.perf_counter()
            metrics = trainer_base.evaluate(
                model=model,
                val_dataloader=val_loader,
                criterion=criterion,
                decision_threshold=cfg["train"]["decision_threshold"],
                device=device,
                binary=binary,
                edge_index=edge_index,
                num_classes=dataset.num_classes,
            )
            synchronize(device)
            validation["actual_seconds"] = time.perf_counter() - start
            validation["metrics"] = [float(value) for value in metrics]
        finally:
            trainer_base.tqdm = original_tqdm

    synchronize(device)
    report = {
        "config": args.config,
        "model": model_name,
        "device": str(device),
        "torch_version": torch.__version__,
        "samples": len(dataset),
        "batch_size": int(cfg["train"]["batch_size"]),
        "num_workers": args.num_workers,
        "non_blocking_transfer": device.type == "cuda",
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "test_batches": len(test_loader),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "edge_shape": list(edge_index.shape),
        "edge_memory_mib": edge_index.numel() * edge_index.element_size() / 2**20,
        "timings_seconds": {
            "dataset": dataset_seconds,
            "split_and_loaders": split_seconds,
            "train_dataframe_slice": train_df_seconds,
            "model_build": model_build_seconds,
            "edge_build": edge_build_seconds,
            "edge_transfer_and_precompute": edge_transfer_seconds,
            "first_batch": first_batch_seconds,
        },
        "training": {**throughput, **phases},
        "validation": validation,
        "device_memory_mib": device_memory_mib(device),
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 2**20,
    }
    return report


def main() -> None:
    args = parse_args()
    report = build_profile(args)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")

    gc.collect()


if __name__ == "__main__":
    main()
