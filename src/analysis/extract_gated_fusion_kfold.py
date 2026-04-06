"""
extract_gated_fusion_kfold.py

Extract GatedFusion weights from k-fold CV models and aggregate by LOS.

For each fold:
  1. Load the best.pt checkpoint
  2. Register a forward hook on model.gated_fusion
  3. Run the (shared) test set through the model
  4. Collect per-sample (w_ad, w_dis, w_merged, LOS)

Then average across folds and save:
  src/analysis/gated_fusion_w_los_kfold.csv
  Columns: LOS, w_ad_mean, w_dis_mean, w_merged_mean  (simple flat CSV)

This output is directly usable by los_group_detection.py.

Usage:
    python src/analysis/extract_gated_fusion_kfold.py \
        --run_name "20260302-143833__ctmp_gin__bs=256__lr=2.00e-04__seed=1__cv=5__test=0.15" \
        --device mps
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.data_processing.splits import holdout_test_split_stratified
from src.models.factory import build_model
from src.utils.device_set import device_set


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract GatedFusion weights from k-fold CV models"
    )
    p.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run directory name under runs/protected/k_fold_CV/",
    )
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--fold",
        type=str,
        default="all",
        help="Folds to use: integer 0-4 or 'all'",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(_THIS_DIR / "gated_fusion_w_los_kfold.csv"),
        help="Output CSV path",
    )
    return p.parse_args()


def load_yaml(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_fold_arg(fold_str: str, n_folds: int) -> list[int]:
    if fold_str.strip().lower() == "all":
        return list(range(n_folds))
    fold_id = int(fold_str.strip())
    if not (0 <= fold_id < n_folds):
        raise ValueError(f"fold {fold_id} out of range [0, {n_folds})")
    return [fold_id]


def _load_fold_model(fold_dir: str, device: torch.device):
    ckpt_path = os.path.join(fold_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    cfg["model"]["params"]["device"] = str(device)

    model = build_model(
        model_name=cfg["model"]["name"], **cfg["model"].get("params", {})
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def extract_weights_one_fold(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """
    Run the test set through the model with a forward hook on model.gated_fusion.
    Returns DataFrame with columns: w_ad, w_dis, w_merged, LOS
    """
    captured_w: list[torch.Tensor] = []

    def _hook(module, input, output):
        # GatedFusion.forward returns (fused, w, logits)
        # output[1] is w: [B, 3]
        _, w, _ = output
        captured_w.append(w.detach().cpu())

    if not hasattr(model, "gated_fusion") or model.gated_fusion is None:
        raise AttributeError(
            "model.gated_fusion not found or is None. "
            "Check that the model was trained with GatedFusion enabled."
        )

    hook_handle = model.gated_fusion.register_forward_hook(_hook)

    all_los: list[torch.Tensor] = []

    try:
        with torch.no_grad():
            for batch in dataloader:
                x, y, los = batch
                x = x.to(device)
                los = los.to(device)

                # edge_index is not in dataloader — inject it via model's stored edge_index
                # We'll pass edge_index separately; caller must set model in eval mode.
                # Note: dataloader yields (x, y, los); the hook fires during forward()
                all_los.append(los.cpu())
    finally:
        hook_handle.remove()

    w_all = torch.cat(captured_w, dim=0).numpy()   # [N, 3]
    los_all = torch.cat(all_los, dim=0).numpy()     # [N]

    df = pd.DataFrame(
        {
            "w_ad": w_all[:, 0],
            "w_dis": w_all[:, 1],
            "w_merged": w_all[:, 2],
            "LOS": los_all.astype(int),
        }
    )
    return df


def extract_weights_one_fold_with_edge(
    model: torch.nn.Module,
    all_x: torch.Tensor,
    all_y: torch.Tensor,
    all_los: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    """
    Run the test set (pre-loaded tensors) through the model with a forward hook.
    Returns DataFrame with columns: w_ad, w_dis, w_merged, LOS
    """
    captured_w: list[torch.Tensor] = []

    def _hook(module, input, output):
        _, w, _ = output
        captured_w.append(w.detach().cpu())

    if not hasattr(model, "gated_fusion") or model.gated_fusion is None:
        raise AttributeError("model.gated_fusion not found or None.")

    hook_handle = model.gated_fusion.register_forward_hook(_hook)
    N = all_x.size(0)

    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                x_b = all_x[start:end].to(device)
                los_b = all_los[start:end].to(device)
                model(x_b, los_b, edge_index, device=device)
    finally:
        hook_handle.remove()

    w_arr = torch.cat(captured_w, dim=0).numpy()   # [N, 3]
    los_arr = all_los.numpy().astype(int)           # [N]

    return pd.DataFrame(
        {
            "w_ad": w_arr[:, 0],
            "w_dis": w_arr[:, 1],
            "w_merged": w_arr[:, 2],
            "LOS": los_arr,
        }
    )


def aggregate_by_los(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate per-fold DataFrames and compute per-LOS mean of (w_ad, w_dis, w_merged).

    Returns flat DataFrame:
        LOS | w_ad_mean | w_dis_mean | w_merged_mean | n_samples
    """
    combined = pd.concat(dfs, ignore_index=True)
    agg = (
        combined.groupby("LOS")[["w_ad", "w_dis", "w_merged"]]
        .agg(["mean", "std", "count"])
    )
    # Flatten multi-level columns: (w_ad, mean) -> w_ad_mean
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg = agg.rename(columns={"w_ad_count": "n_samples"})
    # Drop duplicate count columns
    for col in ["w_dis_count", "w_merged_count"]:
        if col in agg.columns:
            agg = agg.drop(columns=[col])

    return agg.sort_values("LOS").reset_index(drop=True)


def main():
    args = parse_args()

    runs_base = _PROJECT_ROOT / "runs" / "protected" / "k_fold_CV"
    run_dir = runs_base / args.run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    fold_0_cfg_path = run_dir / "folds" / "fold_0" / "config.final.yaml"
    fold_0_cfg = load_yaml(str(fold_0_cfg_path))
    n_folds = fold_0_cfg["train"].get("n_folds", 5)
    folds_to_run = _parse_fold_arg(args.fold, n_folds)

    device = device_set(args.device)
    batch_size = args.batch_size or fold_0_cfg["train"]["batch_size"]

    # Dataset
    data_root = str(_PROJECT_ROOT / "src" / "data")
    dataset = TEDSTensorDataset(
        root=data_root,
        binary=fold_0_cfg["train"].get("binary", True),
        ig_label=fold_0_cfg["train"].get("ig_label", False),
    )

    # Reconstruct test_idx (same for all folds)
    seed = fold_0_cfg["train"]["seed"]
    test_ratio = fold_0_cfg["train"]["test_ratio"]
    _, test_idx = holdout_test_split_stratified(dataset, test_ratio=test_ratio, seed=seed)
    print(f"Test set size: {len(test_idx)}")

    # Pre-load test tensors (CPU)
    print("Pre-loading test tensors...")
    xs, ys, lss = [], [], []
    loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=fold_0_cfg["train"].get("num_workers", 0),
        drop_last=False,
    )
    for x, y, los in loader:
        xs.append(x); ys.append(y); lss.append(los)
    all_x = torch.cat(xs, dim=0)
    all_y = torch.cat(ys, dim=0)
    all_los = torch.cat(lss, dim=0)
    print(f"  all_x: {tuple(all_x.shape)}, all_los range: [{all_los.min()}, {all_los.max()}]")

    # Edge index
    edge_index_path = _PROJECT_ROOT / "src" / "explainers" / "edge_index.pickle"
    with open(edge_index_path, "rb") as f:
        edge_index = pickle.load(f)
    edge_index = edge_index.to(device)

    # Extract per-fold
    fold_dfs: list[pd.DataFrame] = []
    for fold_id in folds_to_run:
        fold_dir = str(run_dir / "folds" / f"fold_{fold_id}")
        print(f"\n=== Fold {fold_id}: extracting GatedFusion weights ===")

        model, _ = _load_fold_model(fold_dir, device)

        df_fold = extract_weights_one_fold_with_edge(
            model=model,
            all_x=all_x,
            all_y=all_y,
            all_los=all_los,
            edge_index=edge_index,
            device=device,
            batch_size=batch_size,
        )
        print(f"  Collected {len(df_fold)} samples, LOS range [{df_fold['LOS'].min()}, {df_fold['LOS'].max()}]")
        print(f"  Mean weights — w_ad={df_fold['w_ad'].mean():.3f}, w_dis={df_fold['w_dis'].mean():.3f}, w_merged={df_fold['w_merged'].mean():.3f}")
        fold_dfs.append(df_fold)

    # Aggregate
    print(f"\n=== Aggregating across {len(fold_dfs)} fold(s) ===")
    df_agg = aggregate_by_los(fold_dfs)
    print(df_agg[["LOS", "w_ad_mean", "w_dis_mean", "w_merged_mean", "n_samples"]].to_string(index=False))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print("→ Now run los_group_detection.py --csv", output_path)


if __name__ == "__main__":
    main()
