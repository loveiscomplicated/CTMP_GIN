"""
permutation_importance_ctmpgin.py

Permutation-based global importance for CTMP-GIN style models where:

Model forward signature:
    model(x, los, edge_index, device) -> logits (or (logits, ...))

Dataloader yields batches unpackable as:
    x, los, y

Edge index is fixed for the whole dataset and provided once to the module.
The module will reuse the fixed edge_index for all batches.

Outputs:
    pandas.DataFrame sorted by importance (descending), including:
      - each node(variable) importance via permuting x[:, j, :] across patients
      - LOS importance via permuting los across patients

Metric:
    Binary ROC-AUC (default).

Progress:
    Uses a single tqdm bar over the total number of permutations:
        total = V * R + R(LOS) = (V + 1) * R
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, Dict
import sys
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

Tensor = torch.Tensor


# -----------------------------
# Config
# -----------------------------

@dataclass
class PermutationImportanceConfig:
    num_repeats: int = 10
    seed: int = 42
    variable_names: Optional[List[str]] = None  # length V if provided


# -----------------------------
# Helpers
# -----------------------------

def _permute_node_feature(x: Tensor, node_idx: int, rng: torch.Generator) -> Tensor:
    """
    Permute a single variable(node) across the batch dimension.

    Supports:
      - x: [B, V]      (categorical ids / scalar features)
      - x: [B, V, F]   (embedded / vector features)
    """
    if x.dim() == 2:
        B = x.size(0)
        perm = torch.randperm(B, generator=rng, device=x.device)
        x_perm = x.clone()
        x_perm[:, node_idx] = x_perm[perm, node_idx]
        return x_perm

    if x.dim() == 3:
        B = x.size(0)
        perm = torch.randperm(B, generator=rng, device=x.device)
        x_perm = x.clone()
        x_perm[:, node_idx, :] = x_perm[perm, node_idx, :]
        return x_perm

    raise ValueError(f"Expected x to have shape [B,V] or [B,V,F], got {tuple(x.shape)}")



def _permute_los(los: Tensor, rng: torch.Generator) -> Tensor:
    """Permute los across batch dimension (dim=0). los: [B] or [B, ...]"""
    if los.dim() < 1:
        raise ValueError(f"Expected los dim >= 1, got {tuple(los.shape)}")
    B = los.size(0)
    perm = torch.randperm(B, generator=rng, device=los.device)
    return los[perm]


def _to_device(t: Tensor, device: Union[str, torch.device]) -> Tensor:
    return t.to(device)


def _predict_proba(
    model: torch.nn.Module,
    x: Tensor,
    los: Tensor,
    edge_index: Tensor,
    device: Union[str, torch.device],
) -> Tensor:
    """
    Calls model(x, los, edge_index, device) and returns prob for class 1.

    Supported logits shapes:
      - [B] or [B, 1] : sigmoid
      - [B, 2]        : softmax -> [:, 1]
    """
    out = model(x, los, edge_index, device=device)
    if isinstance(out, (tuple, list)):
        out = out[0]

    logits = out
    if logits.dim() == 2 and logits.size(-1) == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    if logits.dim() == 2 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    if logits.dim() == 1:
        return torch.sigmoid(logits)

    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def _compute_auc(
    model: torch.nn.Module,
    dataloader: Iterable,
    edge_index: Tensor,
    device: Union[str, torch.device],
    mutate_fn: Optional[Callable[[Tensor, Tensor, torch.Generator], Tuple[Tensor, Tensor]]] = None,
    seed: int = 0,
) -> float:
    """
    Compute ROC-AUC over the dataloader.
    Optionally applies a mutation to (x, los) per batch.

    mutate_fn signature:
        (x, los, rng) -> (x_mut, los_mut)
    """
    model.eval()
    y_all: List[np.ndarray] = []
    p_all: List[np.ndarray] = []

    rng = torch.Generator(device=str(device) if isinstance(device, torch.device) else device)
    rng.manual_seed(seed)

    edge_index = edge_index.to(device)
    with torch.no_grad():   
        for batch in tqdm(dataloader):
            x, y, los = batch
            x = _to_device(x, device)
            y = _to_device(y, device)
            los = _to_device(los, device)

            if mutate_fn is not None:
                x, los = mutate_fn(x, los, rng)

            probs = _predict_proba(model, x, los, edge_index, device=device)

            y_1d = _to_1d_binary_labels(y)
            p_1d = _to_1d_pos_scores(probs)

            y_all.append(y_1d.detach().cpu().numpy())
            p_all.append(p_1d.detach().cpu().numpy())

    y_cat = np.concatenate(y_all, axis=0).reshape(-1)
    p_cat = np.concatenate(p_all, axis=0).reshape(-1)

    if np.unique(y_cat).size < 2:
        return float("nan")
    return float(roc_auc_score(y_cat, p_cat))


# -----------------------------
# Public API
# -----------------------------

def _to_1d_binary_labels(y: Tensor) -> Tensor:
    """
    Ensure y is 1D binary labels [N] with values in {0,1}.
    Accepts:
      - [N]
      - [N,2] one-hot or logits/probs
      - [2,N] transposed one-hot
    """
    if y.dim() == 1:
        return y.long().view(-1)

    if y.dim() == 2:
        # Common cases: [N,2] or [2,N]
        if y.size(1) == 2:          # [N,2]
            return torch.argmax(y, dim=1).long().view(-1)
        if y.size(0) == 2:          # [2,N]
            return torch.argmax(y, dim=0).long().view(-1)

    raise ValueError(f"Unsupported y shape for binary labels: {tuple(y.shape)}")


def _to_1d_pos_scores(p: Tensor) -> Tensor:
    """
    Ensure predicted scores are 1D positive-class scores [N].
    Accepts:
      - [N] (already positive-class prob/logit)
      - [N,2] (class probs/logits)
      - [2,N] (transposed)
    """
    if p.dim() == 1:
        return p.view(-1)

    if p.dim() == 2:
        if p.size(1) == 2:          # [N,2]
            return p[:, 1].contiguous().view(-1)
        if p.size(0) == 2:          # [2,N]
            return p[1, :].contiguous().view(-1)

    raise ValueError(f"Unsupported prediction shape for binary AUC: {tuple(p.shape)}")


def compute_permutation_importance(
    model: torch.nn.Module,
    dataloader: Iterable,
    edge_index: Tensor,
    device: Union[str, torch.device] = "cuda",
    config: Optional[PermutationImportanceConfig] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    if config is None:
        config = PermutationImportanceConfig()

    # ✅ V 먼저 추정 (baseline 전에 해야 total_steps 계산 가능)
    first_batch = next(iter(dataloader))
    x0, y0, los0 = first_batch  # noqa: F841
    if not torch.is_tensor(x0) or x0.dim() not in (2, 3):
        raise ValueError(
            f"Expected x to have shape [B,V] or [B,V,F]. Got {type(x0)} with shape {getattr(x0, 'shape', None)}"
        )
    V = x0.size(1)

    # Variable names
    if config.variable_names is not None:
        if len(config.variable_names) != V:
            raise ValueError(f"variable_names length ({len(config.variable_names)}) must match V ({V}).")
        var_names = config.variable_names
    else:
        var_names = [f"var_{j}" for j in range(V)]

    # ✅ baseline(1회) + (V nodes * R) + (LOS * R)
    total_steps = 1 + (V + 1) * config.num_repeats

    # ✅ stdout 말고 stderr로 강제 출력 (IDE/로그에 묻히는 문제 방지)
    pbar = tqdm(
        total=total_steps,
        desc="Permutation importance",
        disable=not show_progress,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.1,
    )

    try:
        # ✅ Baseline AUC (진행률 포함)
        print("computing baseline auc...")
        baseline_auc = _compute_auc(
            model=model,
            dataloader=dataloader,
            edge_index=edge_index,
            device=device,
            mutate_fn=None,
            seed=config.seed,
        )
        pbar.update(1)
        pbar.set_postfix_str(f"baseline_auc={baseline_auc:.4f}")

        rows: List[Dict[str, Any]] = []

        # Node(variable) importances
        for j in range(V):
            perm_aucs: List[float] = []

            for r in range(config.num_repeats):
                def mutate(x: Tensor, los: Tensor, rng: torch.Generator, node_idx=j) -> Tuple[Tensor, Tensor]:
                    return _permute_node_feature(x, node_idx=node_idx, rng=rng), los

                auc_p = _compute_auc(
                    model=model,
                    dataloader=dataloader,
                    edge_index=edge_index,
                    device=device,
                    mutate_fn=mutate,
                    seed=config.seed + 1000 * j + r,
                )
                perm_aucs.append(auc_p)
                pbar.update(1)

            perm_mean = float(np.nanmean(perm_aucs))
            imp_vals = [
                baseline_auc - a
                for a in perm_aucs
                if not np.isnan(a) and not np.isnan(baseline_auc)
            ]
            imp_mean = float(np.nanmean(imp_vals)) if len(imp_vals) else float("nan")
            imp_std = float(np.nanstd(imp_vals)) if len(imp_vals) else float("nan")

            rows.append(
                dict(
                    feature=var_names[j],
                    kind="node",
                    index=j,
                    baseline_auc=baseline_auc,
                    perm_auc_mean=perm_mean,
                    importance_mean=imp_mean,
                    importance_std=imp_std,
                    num_repeats=config.num_repeats,
                )
            )

        # LOS importance
        perm_aucs_los: List[float] = []
        for r in range(config.num_repeats):
            def mutate_los(x: Tensor, los: Tensor, rng: torch.Generator) -> Tuple[Tensor, Tensor]:
                return x, _permute_los(los, rng=rng)

            auc_p = _compute_auc(
                model=model,
                dataloader=dataloader,
                edge_index=edge_index,
                device=device,
                mutate_fn=mutate_los,
                seed=config.seed + 999999 + r,
            )
            perm_aucs_los.append(auc_p)
            pbar.update(1)

        perm_mean = float(np.nanmean(perm_aucs_los))
        imp_vals = [
            baseline_auc - a
            for a in perm_aucs_los
            if not np.isnan(a) and not np.isnan(baseline_auc)
        ]
        imp_mean = float(np.nanmean(imp_vals)) if len(imp_vals) else float("nan")
        imp_std = float(np.nanstd(imp_vals)) if len(imp_vals) else float("nan")

        rows.append(
            dict(
                feature="LOS",
                kind="los",
                index=None,
                baseline_auc=baseline_auc,
                perm_auc_mean=perm_mean,
                importance_mean=imp_mean,
                importance_std=imp_std,
                num_repeats=config.num_repeats,
            )
        )

    finally:
        pbar.close()

    df = pd.DataFrame(rows)
    df = df.sort_values("importance_mean", ascending=False, na_position="last").reset_index(drop=True)
    return df


if __name__ == "__main__":
    # model = ... (load weights)
    # test_loader = ...  # yields (x, los, y)
    # edge_index = ...   # fixed tensor
    #
    # cfg = PermutationImportanceConfig(num_repeats=10, seed=42, variable_names=[...])
    # df = compute_permutation_importance(model, test_loader, edge_index=edge_index, device="cuda", config=cfg)
    # print(df.head(30))
    pass
