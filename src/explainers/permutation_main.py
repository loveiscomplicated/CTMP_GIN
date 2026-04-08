# permutation_main.py

"""
  ---
  최종 실행 순서

# Step 1: k-fold CV 모델들에서 GatedFusion 가중치 추출
python src/analysis/extract_gated_fusion_kfold.py --run_name "20260302-121934__ctmp_gin__bs=256__lr=2.00e-04__seed=3__cv=5__test=0.15" --device mps
# → src/analysis/gated_fusion_w_los_kfold.csv 생성

# Step 2: LOS 그룹 결정 (k-fold CSV 자동 우선 사용)
python src/analysis/los_group_detection.py
# → src/analysis/los_groups.json + 시각화 생성

# Step 3: PI 실행
python src/explainers/permutation_main.py --run_name "20260302-121934__ctmp_gin__bs=256__lr=2.00e-04__seed=3__cv=5__test=0.15" --fold all --device mps --num_repeats 5 --max_test_samples 30000

"""
import json
import os
import sys
from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

project_root = Path(__file__).resolve().parent.parent.parent  # Phase_2_public/
sys.path.insert(0, str(project_root))

from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.data_processing.splits import (
    holdout_test_split_stratified,
    kfold_stratified,
)
from src.models.factory import build_model, build_edge
from src.utils.device_set import device_set
from src.utils.seed_set import set_seed

from src.explainers.permutation_importance import (
    PermutationImportanceConfig,
    compute_permutation_importance,
)

try:
    from src.explainers.stablity_report import (
        importance_mean_std_table,
    )
except Exception:
    from stablity_report import (
        importance_mean_std_table,
    )


cur_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(cur_dir, "..", "data")
save_base = os.path.join(cur_dir, "results", "permutation")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run directory name under runs/protected/k_fold_CV/",
    )
    p.add_argument(
        "--fold",
        type=str,
        default="all",
        help="Fold to evaluate: integer 0-4 or 'all'",
    )
    p.add_argument(
        "--los_groups",
        type=str,
        default=None,
        help="Path to JSON file with LOS group definitions (optional)",
    )
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--num_repeats", type=int, default=5, help="Permutation repeats per feature"
    )
    p.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Cap test set size for PI (e.g. 30000). Stratified random sample. "
        "None = use full test set.",
    )

    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_fold_arg(fold_str: str, n_folds: int) -> list[int]:
    if fold_str.strip().lower() == "all":
        return list(range(n_folds))
    try:
        fold_id = int(fold_str.strip())
        if not (0 <= fold_id < n_folds):
            raise ValueError(f"fold {fold_id} out of range [0, {n_folds})")
        return [fold_id]
    except ValueError:
        raise ValueError(f"--fold must be an integer or 'all', got: {fold_str!r}")


def _load_fold_model(fold_dir: str, device: torch.device) -> tuple[nn.Module, dict]:
    """Load model from last.pt (matches stored test metrics). cfg is embedded in the checkpoint."""
    ckpt_path = os.path.join(fold_dir, "checkpoints", "last.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    # Override device in model params
    cfg["model"]["params"]["device"] = str(device)

    model = build_model(
        model_name=cfg["model"]["name"], **cfg["model"].get("params", {})
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Loaded checkpoint: epoch={ckpt.get('epoch')}")
    if "metrics" in ckpt and ckpt["metrics"]:
        m = ckpt["metrics"]
        if "val_auc" in m:
            print(f"  val_auc={m['val_auc']:.4f}")

    return model, cfg


def _reconstruct_test_idx(dataset: TEDSTensorDataset, cfg: dict) -> np.ndarray:
    """
    Reproduce the same test_idx used during k-fold CV training.
    holdout_test_split_stratified is called once before the fold loop,
    so test_idx is identical across all folds.
    """
    seed = cfg["train"]["seed"]
    test_ratio = cfg["train"]["test_ratio"]
    _, test_idx = holdout_test_split_stratified(
        dataset, test_ratio=test_ratio, seed=seed
    )
    return test_idx


def _subsample_test_idx(
    test_idx: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """Randomly subsample test_idx to at most max_samples indices."""
    if len(test_idx) <= max_samples:
        return test_idx
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(test_idx), size=max_samples, replace=False)
    return test_idx[np.sort(chosen)]


def _build_fold_edge_index(
    dataset: TEDSTensorDataset,
    cfg: dict,
    fold_id: int,
    device: torch.device,
    data_root: str,
    fold_dir: str | None = None,
) -> torch.Tensor:
    """
    Load saved edge_index.pt if available; otherwise reconstruct from training data.

    Mirrors run_kfold_cv.py:
        trainval_idx, test_idx = holdout_test_split_stratified(...)
        for fold, train_idx, val_idx in kfold_stratified(...):
            train_df = dataset.processed_df.iloc[train_idx]
            edge_index = build_edge(model_name, root, seed, train_df, ...)
    """
    # Load saved edge_index if available
    if fold_dir is not None:
        edge_index_path = os.path.join(fold_dir, "edge_index.pt")
        if os.path.exists(edge_index_path):
            print(f"  Loading saved edge_index from {edge_index_path}")
            return torch.load(edge_index_path, map_location=device)
        print(f"  edge_index.pt not found, recomputing...")

    seed = cfg["train"]["seed"]
    test_ratio = cfg["train"]["test_ratio"]
    n_folds = cfg["train"]["n_folds"]
    model_name = cfg["model"]["name"]
    batch_size = cfg["train"]["batch_size"]

    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    trainval_idx, _ = holdout_test_split_stratified(
        dataset, test_ratio=test_ratio, seed=seed, labels=labels
    )

    # num_nodes: matches run_kfold_cv.py logic
    if model_name == "gin":
        num_nodes = len(dataset.col_info[0])
    else:
        num_nodes = len(dataset.col_info[2])  # ad_col_index length

    train_idx = None
    for fold, t_idx, _ in kfold_stratified(
        trainval_idx=trainval_idx, labels=labels, n_folds=n_folds, seed=seed
    ):
        if fold == fold_id:
            train_idx = t_idx
            break

    if train_idx is None:
        raise ValueError(f"fold_id={fold_id} not found in kfold_stratified")

    train_df = dataset.processed_df.iloc[train_idx]

    set_seed(seed)
    result = build_edge(
        model_name=model_name,
        root=data_root,
        seed=seed,
        train_df=train_df,
        num_nodes=num_nodes,
        batch_size=batch_size,
        **cfg.get("edge", {}),
    )
    # build_edge may return (edge_index, edge_attr) or just edge_index
    edge_index = result[0] if isinstance(result, tuple) else result
    return edge_index.to(device)


def _build_test_loader(
    dataset: TEDSTensorDataset,
    test_idx,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    # drop_last=False: we need ALL test samples for PI evaluation
    indices = test_idx.tolist() if hasattr(test_idx, "tolist") else list(test_idx)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
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

    # ---- Resolve paths ----
    project_root_dir = Path(cur_dir).resolve().parent.parent
    runs_base = project_root_dir / "runs" / "protected" / "k_fold_CV"
    run_dir = runs_base / args.run_name

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Read fold_0 config for shared settings (seed, test_ratio, n_folds, etc.)
    fold_0_cfg_path = run_dir / "folds" / "fold_0" / "config.final.yaml"
    fold_0_cfg = load_yaml(str(fold_0_cfg_path))
    n_folds = fold_0_cfg["train"].get("n_folds", 5)

    folds_to_run = _parse_fold_arg(args.fold, n_folds)
    print(f"Run: {args.run_name}")
    print(f"Folds to evaluate: {folds_to_run}")

    # ---- Device ----
    device = device_set(args.device)

    # ---- Dataset (shared across folds) ----
    model_name = fold_0_cfg["model"]["name"]
    remove_los = model_name not in ["gin", "a3tgcn_2_points", "gin_gru_2_points"]
    dataset = TEDSTensorDataset(
        root=root,
        binary=fold_0_cfg["train"].get("binary", True),
        ig_label=fold_0_cfg["train"].get("ig_label", False),
        remove_los=remove_los,
        do_preprocess=fold_0_cfg["train"].get("do_preprocess", True),
    )
    col_names = dataset.col_info[0]
    V = len(col_names)
    names_with_los = col_names + ["LOS"]

    # ---- Reconstruct test_idx (same for all folds) ----
    test_idx = _reconstruct_test_idx(dataset, fold_0_cfg)
    print(f"Test set size (full): {len(test_idx)}")
    if args.max_test_samples is not None:
        test_idx = _subsample_test_idx(
            test_idx, args.max_test_samples, seed=fold_0_cfg["train"]["seed"]
        )
        print(f"Test set size (subsampled): {len(test_idx)}")

    # ---- Optional LOS groups ----
    los_groups = None
    if args.los_groups:
        with open(args.los_groups, "r") as f:
            los_groups = json.load(f)
        assert isinstance(los_groups, dict)
        print(f"LOS groups loaded: {list(los_groups.keys())}")

    # ---- Output directory ----
    out_dir = Path(save_base) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Per-fold PI computation ----
    fold_vecs: list[torch.Tensor] = []
    # group_vecs[gname] = list of importance vectors, one per fold
    group_vecs: dict[str, list[torch.Tensor]] = {}
    if los_groups is not None:
        group_vecs = {gname: [] for gname in los_groups}

    for fold_id in folds_to_run:
        fold_dir = str(run_dir / "folds" / f"fold_{fold_id}")
        print(f"\n=== Fold {fold_id} ===")

        model, fold_cfg = _load_fold_model(fold_dir, device)

        batch_size = args.batch_size or fold_cfg["train"]["batch_size"]
        num_workers = fold_cfg["train"].get("num_workers", 0)

        # Build fold-specific edge_index (load saved if available)
        print(f"  Building edge_index for fold {fold_id}...")
        edge_index = _build_fold_edge_index(
            dataset=dataset,
            cfg=fold_cfg,
            fold_id=fold_id,
            device=device,
            data_root=root,
            fold_dir=fold_dir,
        )
        print(
            f"  edge_index shape: {tuple(edge_index.shape)}, max node: {edge_index.max().item()}"
        )

        test_loader = _build_test_loader(dataset, test_idx, batch_size, num_workers)

        perm_cfg = PermutationImportanceConfig(
            num_repeats=args.num_repeats,
            seed=fold_cfg["train"]["seed"],
            variable_names=col_names,
        )

        df = compute_permutation_importance(
            model=model,
            dataloader=test_loader,
            edge_index=edge_index,
            device=device,
            config=perm_cfg,
        )

        out_csv = out_dir / f"fold_{fold_id}_pi.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
        print(f"  baseline_auc = {df['baseline_auc'].iloc[0]:.4f}")
        print(f"  Top 5 features:")
        print(
            df[["feature", "importance_mean", "importance_std"]]
            .head(5)
            .to_string(index=False)
        )

        fold_vecs.append(df_to_importance_vector(df, V=V))

        # ---- LOS group sub-analysis (optional) ----
        if los_groups is not None:
            los_tensor = dataset.LOS

            for gname, los_vals in los_groups.items():
                los_set = set(los_vals)
                group_idx = [
                    i for i in test_idx if int(round(float(los_tensor[i]))) in los_set
                ]
                if not group_idx:
                    print(
                        f"  Warning: no test samples for {gname} (LOS={los_vals}), skipping"
                    )
                    continue

                print(f"\n  --- {gname} (LOS={los_vals}, n={len(group_idx)}) ---")
                g_loader = _build_test_loader(
                    dataset, group_idx, batch_size, num_workers
                )

                df_g = compute_permutation_importance(
                    model=model,
                    dataloader=g_loader,
                    edge_index=edge_index,
                    device=device,
                    config=perm_cfg,
                    show_progress=False,
                )

                out_g_csv = out_dir / f"fold_{fold_id}_{gname}_pi.csv"
                df_g.to_csv(out_g_csv, index=False)
                print(f"  Saved: {out_g_csv}")

                group_vecs[gname].append(df_to_importance_vector(df_g, V=V))

    # ---- Cross-fold aggregate: overall ----
    if len(fold_vecs) > 1:
        print("\n=== Cross-fold mean±std (overall) ===")
        df_ms = importance_mean_std_table(
            fold_vecs,
            names_with_los,
            save_path=str(out_dir),
            filename="all_folds_pi.csv",
        )
        print("\nTop 30:")
        print(df_ms.head(30).to_string(index=False))
    elif len(fold_vecs) == 1:
        single_fold_id = folds_to_run[0]
        df_single = pd.read_csv(out_dir / f"fold_{single_fold_id}_pi.csv")
        print(f"\n=== Fold {single_fold_id} Top 30 ===")
        print(df_single.head(30).to_string(index=False))

    # ---- Cross-fold aggregate: per group ----
    if los_groups is not None:
        for gname, vecs in group_vecs.items():
            if len(vecs) > 1:
                print(f"\n=== Cross-fold mean±std: {gname} ===")
                df_g_ms = importance_mean_std_table(
                    vecs,
                    names_with_los,
                    save_path=str(out_dir),
                    filename=f"{gname}_all_folds_pi.csv",
                )
                print(f"  Top 10:")
                print(df_g_ms.head(10).to_string(index=False))
            elif len(vecs) == 1:
                print(f"\n=== {gname} (single fold) Top 10 ===")
                print(
                    pd.read_csv(out_dir / f"fold_{folds_to_run[0]}_{gname}_pi.csv")
                    .head(10)
                    .to_string(index=False)
                )


if __name__ == "__main__":
    main()
