# permutation_main.py
import os
import sys
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root.parent))

from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.data_processing.data_utils import train_test_split_stratified
from src.models.factory import build_model
from src.trainers.base import load_checkpoint
from src.utils.seed_set import set_seed
from src.utils.device_set import device_set
from src.trainers.utils.early_stopper import EarlyStopper

from src.explainers.permutation_importance import (
    PermutationImportanceConfig,
    compute_permutation_importance,
)

# stability utils: src에 있으면 그걸 쓰고, 없으면 같은 폴더의 stablity_report.py 사용
try:
    from src.explainers.stablity_report import (
        stability_report,
        print_stability_report,
        unstable_variables_report,
        print_unstable_report_with_names,
        importance_mean_std_table,
    )
except Exception:
    from stablity_report import (
        stability_report,
        print_stability_report,
        unstable_variables_report,
        print_unstable_report_with_names,
        importance_mean_std_table,
    )


cur_dir = os.path.dirname(__file__)
root = os.path.join(cur_dir, "..", "data")
save_path = os.path.join(cur_dir, "results")
os.makedirs(save_path, exist_ok=True)

# --------@@@@ adjust model path !!! @@@@--------
model_path = os.path.join(cur_dir, "..", "..", "runs", "temp_ctmp_gin_ckpt", "ctmp_epoch_36_loss_0.2738.pth")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)

    # 기존 override 옵션들
    p.add_argument("--is_mi_based_edge", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--decision_threshold", type=float, default=None)
    p.add_argument("--binary", type=int, default=None)

    # ✅ sampling + permutation config
    p.add_argument("--sample_ratio", type=float, default=0.1, help="test set sampling ratio per seed (e.g., 0.05)")
    p.add_argument("--max_samples", type=int, default=None, help="cap sampled test examples (after ratio)")
    p.add_argument("--seeds", type=str, default="0,1,2", help="comma-separated seeds for stability (e.g., 0,1,2)")
    p.add_argument("--num_repeats", type=int, default=5, help="permutation repeats per feature")
    p.add_argument("--save_prefix", type=str, default="permutation", help="prefix for saved csv files")

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


def _parse_seeds(seed_str: str) -> list[int]:
    items = [s.strip() for s in seed_str.split(",") if s.strip()]
    return [int(x) for x in items]


def build_sampled_test_loader(
    dataset: torch.utils.data.Dataset,
    test_indices: list[int],
    sample_ratio: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    max_samples: int | None = None,
) -> DataLoader:
    n_total = len(test_indices)
    n = int(n_total * sample_ratio)
    n = max(n, 1)
    if max_samples is not None:
        n = min(n, max_samples)

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    chosen = [test_indices[i] for i in perm[:n].tolist()]

    subset = Subset(dataset, chosen)

    # permutation_importance는 shuffle=False여도 됨 (AUC 평가이므로)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def df_to_importance_vector(df, V: int) -> torch.Tensor:
    """
    df: output of compute_permutation_importance
    returns vector of length (V + 1): [var0..var(V-1), LOS]
    """
    vec = torch.full((V + 1,), float("nan"), dtype=torch.float32)

    for _, row in df.iterrows():
        kind = row.get("kind", None)
        if kind == "node":
            j = int(row["index"])
            vec[j] = float(row["importance_mean"])
        elif kind == "los":
            vec[V] = float(row["importance_mean"])

    return vec


def main():
    args = parse_args()
    cfg = override_cfg(load_yaml(args.config), args)

    # seed (dataset split seed)
    split_seed = cfg["train"].get("seed", 42)
    set_seed(split_seed)

    device = device_set(cfg["device"])

    # dataset
    dataset = TEDSTensorDataset(
        root=root,
        binary=cfg["train"].get("binary", True),
        ig_label=cfg["train"].get("ig_label", False),
    )
    cfg["model"]["params"]["col_info"] = dataset.col_info
    cfg["model"]["params"]["num_classes"] = dataset.num_classes

    # split
    split_ratio = [cfg["train"]["train_ratio"], cfg["train"]["val_ratio"], cfg["train"]["test_ratio"]]
    train_loader, val_loader, test_loader, idx = train_test_split_stratified(
        dataset=dataset,
        batch_size=cfg["train"]["batch_size"],
        ratio=split_ratio,
        seed=split_seed,
        num_workers=cfg["train"]["num_workers"],
    )
    train_idx, val_idx, test_idx = idx  # type: ignore

    # model
    model = build_model(model_name=cfg["model"]["name"], **cfg["model"].get("params", {})).to(device)

    if cfg["train"]["binary"]:
        _ = nn.BCEWithLogitsLoss()
    else:
        _ = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=cfg["train"]["lr_scheduler_patience"])
    _ = EarlyStopper(patience=cfg["train"]["early_stopping_patience"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    start_epoch, best_loss = load_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        filename=model_path,
        map_location=device,
    )
    model.eval()
    print(f"Loaded checkpoint from {model_path}")
    print(f"Checkpoint start_epoch: {start_epoch}, best_loss: {best_loss}")

    # edge_index (fixed)
    import pickle
    with open(os.path.join(cur_dir, "edge_index.pickle"), "rb") as f:
        edge_index = pickle.load(f)
    edge_index = edge_index.to(device)

    # 변수 이름 + LOS 이름
    col_names = dataset.col_info[0]
    V = len(col_names)
    names_with_los = col_names + ["LOS"]

    # ✅ seed별로: (1) test subset 샘플링 -> (2) permutation importance -> (3) vector 저장
    seeds = _parse_seeds(args.seeds)
    outs: list[torch.Tensor] = []
    dfs = []

    for s in seeds:
        print(f"\n=== Permutation on sampled test set | seed={s} | sample_ratio={args.sample_ratio} ===")
        set_seed(s)

        sampled_loader = build_sampled_test_loader(
            dataset=dataset,
            test_indices=list(test_idx),
            sample_ratio=args.sample_ratio,
            seed=s,
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["train"]["num_workers"],
            max_samples=args.max_samples,
        )

        perm_cfg = PermutationImportanceConfig(
            num_repeats=args.num_repeats,
            seed=s,
            variable_names=col_names,
        )

        df = compute_permutation_importance(
            model=model,
            dataloader=sampled_loader,
            edge_index=edge_index,
            device=device,
            config=perm_cfg,
        )

        out_csv = os.path.join(save_path, f"{args.save_prefix}_seed{s}_ratio{args.sample_ratio}.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

        vec = df_to_importance_vector(df, V=V)  # [V+1]
        outs.append(vec)
        dfs.append(df)

    # ✅ stability report (permutation vector 기준)
    print("\n=== Building mean±std table across seeds ===")
    df_ms = importance_mean_std_table(outs, names_with_los)
    out_ms_csv = os.path.join(save_path, f"{args.save_prefix}_mean_std_ratio{args.sample_ratio}.csv")
    df_ms.to_csv(out_ms_csv, index=False)
    print(f"Saved: {out_ms_csv}")

    print("\n=== Top 30 (mean±std) ===")
    print(df_ms.head(30).to_string(index=False))

    report = stability_report(outs, ks=[10, 20, 30])
    print_stability_report(report, ks=[10, 20, 30])

    rep20 = unstable_variables_report(outs, k=20)
    print_unstable_report_with_names(rep20, names_with_los)

    rep30 = unstable_variables_report(outs, k=30)
    print_unstable_report_with_names(rep30, names_with_los)


if __name__ == "__main__":
    main()
