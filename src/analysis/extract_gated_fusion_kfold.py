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
    --run_name "(final)20260413-071956__ctmp_gin__bs=1024__lr=6.10e-04__seed=1__cv=5__test=0.15" \
    --device mps
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.data_processing.splits import holdout_test_split_stratified, kfold_stratified
from src.models.factory import build_model, build_edge
from src.utils.device_set import device_set
from src.utils.seed_set import set_seed


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


def _load_fold_model(fold_dir: str, device: torch.device, best_or_last: str = "best"):
    if best_or_last == "best":
        ckpt_path = os.path.join(fold_dir, "checkpoints", "best.pt")
    else:  # best_or_last == "last"
        ckpt_path = os.path.join(fold_dir, "checkpoints", "last.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    cfg["model"]["params"]["device"] = str(device)

    model = build_model(
        model_name=cfg["model"]["name"], **cfg["model"].get("params", {})
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
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

    w_all = torch.cat(captured_w, dim=0).numpy()  # [N, 3]
    los_all = torch.cat(all_los, dim=0).numpy()  # [N]

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
    Run the test set (pre-loaded tensors) through the model using return_internals=True.
    Returns DataFrame with columns: w_ad, w_dis, w_merged, LOS
    """
    if not hasattr(model, "gated_fusion") or model.gated_fusion is None:
        raise AttributeError("model.gated_fusion not found or None.")

    N = all_x.size(0)
    all_w: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_b = all_x[start:end].to(device)
            los_b = all_los[start:end].to(device)
            _, _, w, _, _, _, _ = model(x_b, los_b, edge_index, return_internals=True)
            all_w.append(w.detach().cpu())

    w_arr = torch.cat(all_w, dim=0).numpy()  # [N, 3]
    los_arr = all_los.numpy().astype(int)  # [N]

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
    agg = combined.groupby("LOS")[["w_ad", "w_dis", "w_merged"]].agg(
        ["mean", "std", "count"]
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


def _sanity_check(
    model: torch.nn.Module,
    all_x: torch.Tensor,
    all_los: torch.Tensor,
    all_y: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    batch_size: int,
    fold_id: int,
    run_dir: Path,
):
    """Compute test AUC/ACC and compare against stored cv_summary.json metrics."""
    model.eval()
    all_logits = []
    N = all_x.size(0)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            logits = model(
                all_x[start:end].to(device),
                all_los[start:end].to(device),
                edge_index,
                device=device,
            )
            all_logits.append(logits.cpu())
    logits_cat = torch.cat(all_logits, dim=0).squeeze(-1)  # [N]
    scores = torch.sigmoid(logits_cat).numpy()
    preds = (scores >= 0.5).astype(int)
    targets = all_y.numpy().astype(int)

    auc = roc_auc_score(targets, scores)
    acc = accuracy_score(targets, preds)

    # Load stored metrics from cv_summary.json
    summary_path = run_dir / "cv_summary.json"
    stored_auc, stored_acc = None, None
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        for fr in summary.get("fold_results", []):
            if fr.get("fold") == fold_id:
                stored_auc = fr.get("test_auc")
                stored_acc = fr.get("test_acc")
                break

    # --- quick diagnostic ---
    print(
        f"  [Sanity] scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}"
    )
    print(
        f"  [Sanity] targets: pos_rate={targets.mean():.4f} ({targets.sum()}/{len(targets)})"
    )
    print(f"  [Sanity] preds:   pos_rate={preds.mean():.4f}")
    # -------------------------
    print(f"  [Sanity] recomputed  — AUC={auc:.6f}, ACC={acc:.6f}")
    if stored_auc is not None and stored_acc is not None:
        auc_delta = abs(auc - stored_auc)
        acc_delta = abs(acc - stored_acc)
        flag = " *** MISMATCH ***" if auc_delta > 0.005 else ""
        print(f"  [Sanity] stored      — AUC={stored_auc:.6f}, ACC={stored_acc:.6f}")
        print(
            f"  [Sanity] delta       — ΔAUC={auc_delta:.6f}, ΔACC={acc_delta:.6f}{flag}"
        )


def _get_train_df_for_fold(
    dataset: TEDSTensorDataset,
    labels: np.ndarray,
    fold_cfg: dict,
    target_fold_id: int,
) -> pd.DataFrame:
    """Reconstruct the train_df used for a specific fold during training."""
    seed = fold_cfg["train"]["seed"]
    test_ratio = fold_cfg["train"]["test_ratio"]
    n_folds = fold_cfg["train"]["n_folds"]

    trainval_idx, _ = holdout_test_split_stratified(
        dataset, test_ratio=test_ratio, seed=seed, labels=labels
    )
    for fold, train_idx, _ in kfold_stratified(
        trainval_idx=trainval_idx, labels=labels, n_folds=n_folds, seed=seed
    ):
        if fold == target_fold_id:
            return dataset.processed_df.iloc[train_idx]

    raise ValueError(f"fold {target_fold_id} not found in {n_folds}-fold split")


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

    # Dataset — mirror run_kfold_cv.py: respect remove_los and do_preprocess
    data_root = str(_PROJECT_ROOT / "src" / "data")
    model_name = fold_0_cfg["model"]["name"]
    remove_los = model_name not in ["gin", "a3tgcn_2_points", "gin_gru_2_points"]
    dataset = TEDSTensorDataset(
        root=data_root,
        binary=fold_0_cfg["train"].get("binary", True),
        ig_label=fold_0_cfg["train"].get("ig_label", False),
        remove_los=remove_los,
        do_preprocess=fold_0_cfg["train"].get("do_preprocess", True),
    )

    # num_nodes (same logic as run_kfold_cv.py)
    if model_name == "gin":
        num_nodes = len(dataset.col_info[0])
    else:
        num_nodes = len(dataset.col_info[2])  # ad_col_index

    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # Reconstruct test_idx (same for all folds, seed-deterministic)
    seed = fold_0_cfg["train"]["seed"]
    test_ratio = fold_0_cfg["train"]["test_ratio"]
    _, test_idx = holdout_test_split_stratified(
        dataset, test_ratio=test_ratio, seed=seed, labels=labels
    )
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
        xs.append(x)
        ys.append(y)
        lss.append(los)
    all_x = torch.cat(xs, dim=0)
    all_y = torch.cat(ys, dim=0)
    all_los = torch.cat(lss, dim=0)
    print(
        f"  all_x: {tuple(all_x.shape)}, all_los range: [{all_los.min()}, {all_los.max()}]"
    )

    # Extract per-fold
    fold_dfs: list[pd.DataFrame] = []
    for fold_id in folds_to_run:
        fold_dir = str(run_dir / "folds" / f"fold_{fold_id}")
        fold_cfg = load_yaml(
            str(run_dir / "folds" / f"fold_{fold_id}" / "config.final.yaml")
        )
        print(f"\n=== Fold {fold_id}: building edge_index ===")

        # Load saved edge_index if available; otherwise recompute
        edge_index_path = os.path.join(fold_dir, "edge_index.pt")
        if os.path.exists(edge_index_path):
            print(f"  Loading saved edge_index from {edge_index_path}")
            edge_index = torch.load(edge_index_path, map_location=device)
        else:
            print(f"  edge_index.pt not found, recomputing...")
            train_df = _get_train_df_for_fold(dataset, labels, fold_cfg, fold_id)
            edge_cfg = fold_cfg.get("edge", {})
            set_seed(fold_cfg["train"]["seed"])
            built = build_edge(
                model_name=model_name,
                root=data_root,
                seed=fold_cfg["train"]["seed"],
                train_df=train_df,
                num_nodes=num_nodes,
                batch_size=fold_cfg["train"]["batch_size"],
                **edge_cfg,
            )
            edge_index = (built[0] if isinstance(built, tuple) else built).to(device)
        print(f"  edge_index shape: {tuple(edge_index.shape)}")

        print(f"=== Fold {fold_id}: extracting GatedFusion weights ===")
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
        print(
            f"  Collected {len(df_fold)} samples, LOS range [{df_fold['LOS'].min()}, {df_fold['LOS'].max()}]"
        )
        print(
            f"  Mean weights — w_ad={df_fold['w_ad'].mean():.3f}, w_dis={df_fold['w_dis'].mean():.3f}, w_merged={df_fold['w_merged'].mean():.3f}"
        )

        # Sanity check: recompute test AUC/ACC and compare to stored metrics
        _sanity_check(
            model,
            all_x,
            all_los,
            all_y,
            edge_index,
            device,
            batch_size,
            fold_id,
            run_dir,
        )

        fold_dfs.append(df_fold)

    # Aggregate
    print(f"\n=== Aggregating across {len(fold_dfs)} fold(s) ===")
    df_agg = aggregate_by_los(fold_dfs)
    print(
        df_agg[
            ["LOS", "w_ad_mean", "w_dis_mean", "w_merged_mean", "n_samples"]
        ].to_string(index=False)
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print("→ Now run los_group_detection.py --csv", output_path)


if __name__ == "__main__":
    main()
