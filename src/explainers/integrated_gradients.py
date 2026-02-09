import torch
from typing import Dict, Optional

@torch.no_grad()
def _make_baseline_x_like(x_idx: torch.Tensor) -> torch.Tensor:
    # baseline category = 0
    return torch.zeros_like(x_idx, dtype=torch.long)

@torch.no_grad()
def _make_baseline_los_like(los_idx: torch.Tensor) -> torch.Tensor:
    # baseline LOS token = 0 (NONE)
    return torch.zeros_like(los_idx, dtype=torch.long)


def manual_ig_embeddings(
    model,
    x_idx: torch.Tensor,          # [B, 72] long
    los_idx: torch.Tensor,        # [B] long
    edge_index: torch.Tensor,     # [2, E_internal]
    target: str = "logit",        # "logit" only for now
    n_steps: int = 50,
    return_full: bool = False,    # True면 ig_x([B,72,D])/ig_los([B,Dlos])도 같이 반환
) -> Dict[str, torch.Tensor]:
    """
    Requires model has:
      - entity_embedding_layer(x_idx) -> [B,72,emb_dim]
      - get_edge_index_2(edge_index, num_nodes, batch_size) -> [2, E_total]
      - get_edge_attr(los, edge_index, batch_size, num_nodes) -> [E_total, D_los]
      - forward_from_x_emb(x_embedded, los, edge_index) -> [B,1] or [B]
      - forward_from_x_emb_with_edge_attr(x_embedded, edge_index, edge_index_2, edge_attr) -> [B,1] or [B]
    """
    assert target == "logit", "manual_ig_embeddings: target='logit' only."

    model.eval()
    device = next(model.parameters()).device

    x_idx = x_idx.to(device).long()
    los_idx = los_idx.to(device).long()
    edge_index = edge_index.to(device)

    B = x_idx.size(0)
    num_nodes = len(model.ad_col_index)  # CTMPGIN 기준

    # -------------------------
    # Baselines (indices)
    # -------------------------
    base_x_idx = _make_baseline_x_like(x_idx).to(device)
    base_los_idx = _make_baseline_los_like(los_idx).to(device)

    # -------------------------
    # Continuous embeddings (x)
    # -------------------------
    x_emb = model.entity_embedding_layer(x_idx)         # [B,72,Dx]
    base_x_emb = model.entity_embedding_layer(base_x_idx)

    def f_from_xemb(xemb: torch.Tensor, los_indices: torch.Tensor) -> torch.Tensor:
        out = model.forward_from_x_emb(xemb, los_indices, edge_index)
        return out.squeeze(-1)  # [B]

    # ==========================================================
    # 1) IG for variables in embedding space: IG over x_emb
    # ==========================================================
    dx = (x_emb - base_x_emb)
    total_grads_x = torch.zeros_like(x_emb)

    for s in range(1, n_steps + 1):
        alpha = float(s) / n_steps
        x_s = (base_x_emb + alpha * dx).detach().requires_grad_(True)

        out = f_from_xemb(x_s, los_idx)  # [B]
        grads = torch.autograd.grad(out.sum(), x_s, retain_graph=False, create_graph=False)[0]
        total_grads_x += grads

    avg_grads_x = total_grads_x / float(n_steps)
    ig_x = dx * avg_grads_x  # [B,72,Dx]

    # ==========================================================
    # 2) IG for LOS via edge_attr path: IG over edge_attr
    #    (edge_attr는 [E_total, D_los] 이고, cross-edge만 sample별로 모음)
    # ==========================================================
    # Build fixed edge_index_2 once
    edge_index_2 = model.get_edge_index_2(
        edge_index=edge_index, num_nodes=num_nodes, batch_size=B
    ).to(device)

    # Actual / baseline edge_attr
    edge_attr_1 = model.get_edge_attr(
        los=los_idx, edge_index=edge_index, batch_size=B, num_nodes=num_nodes
    ).to(device)

    edge_attr_0 = model.get_edge_attr(
        los=torch.zeros_like(los_idx), edge_index=edge_index, batch_size=B, num_nodes=num_nodes
    ).to(device)

    d_edge = (edge_attr_1 - edge_attr_0)
    total_grads_ea = torch.zeros_like(edge_attr_1)

    # x는 actual로 고정한 상태에서 LOS만 IG
    x_emb_det = x_emb.detach()

    for s in range(1, n_steps + 1):
        alpha = float(s) / n_steps
        ea_s = (edge_attr_0 + alpha * d_edge).detach().requires_grad_(True)

        out = model.forward_from_x_emb_with_edge_attr(
            x_embedded=x_emb_det,
            edge_index=edge_index,
            edge_index_2=edge_index_2,
            edge_attr=ea_s,
        ).squeeze(-1)  # [B]

        grads = torch.autograd.grad(out.sum(), ea_s, retain_graph=False, create_graph=False)[0]
        total_grads_ea += grads

    avg_grads_ea = total_grads_ea / float(n_steps)
    ig_edge_attr = d_edge * avg_grads_ea  # [E_total, D_los]

    # Cross edges만 모아 per-sample LOS attribution 만들기
    E_cross = B * num_nodes
    E_total = ig_edge_attr.size(0)
    cross = ig_edge_attr[E_total - E_cross:]                 # [B*N, D_los]
    ig_los = cross.reshape(B, num_nodes, -1).sum(dim=1)      # [B, D_los]

    # ==========================================================
    # 3) Reduce to variable-level scores (abs / signed)
    # ==========================================================
    imp_abs_x = ig_x.abs().sum(dim=-1)       # [B,72]
    imp_signed_x = ig_x.sum(dim=-1)          # [B,72]

    imp_abs_los = ig_los.abs().sum(dim=-1)   # [B]
    imp_signed_los = ig_los.sum(dim=-1)      # [B]

    # ==========================================================
    # 4) Completeness-ish delta (SIGNED로만)
    # ==========================================================
    with torch.no_grad():
        fx = f_from_xemb(x_emb, los_idx)              # [B]
        f0 = f_from_xemb(base_x_emb, base_los_idx)    # [B]

        ig_x_scalar = imp_signed_x.sum(dim=1)         # [B]
        ig_los_scalar = imp_signed_los                # [B]
        delta = (fx - f0) - (ig_x_scalar + ig_los_scalar)  # [B]

    out = {
        "imp_abs_x": imp_abs_x,               # [B,72]
        "imp_signed_x": imp_signed_x,         # [B,72]
        "imp_abs_los": imp_abs_los,           # [B]
        "imp_signed_los": imp_signed_los,     # [B]
        "delta": delta,                       # [B]
    }

    if return_full:
        out["ig_x"] = ig_x                   # [B,72,Dx]
        out["ig_los"] = ig_los               # [B,D_los]
        out["ig_edge_attr"] = ig_edge_attr   # [E_total,D_los]

    return out


from tqdm import tqdm
from src.explainers.utils import _iter_selected_batches
from typing import Optional, List, Literal


"""
      ig_x_emb:  [B, 72, emb_dim]  (variable attribution in embedding space)
      ig_los_emb:[B, los_emb_dim]  (LOS attribution in embedding space)
      delta:     [B] (approx convergence check: f(x)-f(baseline) - sum(IG))
"""


def explain_ig_for_dataset(
    dataloader,
    model,
    edge_index: torch.Tensor,     # [2, E]
    target: str = "logit",        # "logit" only for now
    n_steps: int = 50,
    reduce: Literal["mean", "median"] = "mean",
    keep_all: bool = False,
    max_batches: Optional[int] = None,
    verbose: bool = True,
    sample_ratio: float = 0.1,
    seed: int = 0,
    ):

    model.eval()

    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1].")

    # --- storage ---
    scores_abs_x: List[torch.Tensor] = []
    scores_abs_los: List[torch.Tensor] = []
    scores_signed_x: List[torch.Tensor] = []
    scores_signed_los: List[torch.Tensor] = []
    scores_delta: List[torch.Tensor] = []

    running_sum_abs_x: Optional[torch.Tensor] = None
    running_sum_abs_los: Optional[torch.Tensor] = None
    running_sum_signed_x: Optional[torch.Tensor] = None
    running_sum_signed_los: Optional[torch.Tensor] = None
    running_sum_delta: Optional[torch.Tensor] = None

    n_seen = 0

    if not hasattr(dataloader, "__len__"):
        raise ValueError("Sampling requires dataloader with __len__().")

    total_batches = len(dataloader)
    effective_batches = min(total_batches, max_batches) if max_batches is not None else total_batches

    if sample_ratio < 1.0:
        g = torch.Generator().manual_seed(seed)
        n_pick = max(1, int(round(effective_batches * sample_ratio)))
        perm = torch.randperm(effective_batches, generator=g).tolist()
        selected = sorted(perm[:n_pick])
        if verbose:
            print(f"[Integrated Gradients] Sampling batches: {n_pick}/{effective_batches} (ratio={sample_ratio}, seed={seed})")
    else:
        selected = list(range(effective_batches))

    iterator = _iter_selected_batches(dataloader, selected)
    pbar = tqdm(iterator, total=len(selected), desc="IG (dataset)")

    eps = 1e-12  # safeguard for rare zero-total cases

    for b_idx, batch in pbar:
        x_idx, y_idx, los_idx = batch

        res = manual_ig_embeddings(
            model=model,
            x_idx=x_idx,
            los_idx=los_idx,
            edge_index=edge_index,
            target=target,
            n_steps=n_steps,
        )

        imp_abs_x = res["imp_abs_x"]           # [B,72]
        imp_abs_los = res["imp_abs_los"]       # [B]
        imp_signed_x = res["imp_signed_x"]     # [B,72] (leave as-is)
        imp_signed_los = res["imp_signed_los"] # [B]    (leave as-is)
        delta = res["delta"]                   # [B]    (leave as-is)

        # -----------------------------
        # NEW: per-sample share (ABS only)
        # -----------------------------
        # total_abs per sample: [B]
        total_abs = imp_abs_x.sum(dim=1) + imp_abs_los
        total_abs = total_abs + eps  # avoid divide-by-zero

        share_x = imp_abs_x / total_abs.unsqueeze(1)  # [B,72]
        share_los = imp_abs_los / total_abs           # [B]

        B = share_x.size(0)
        n_seen += B

        if reduce == "mean" and not keep_all:
            # NOTE: we now accumulate shares instead of raw abs importances
            if running_sum_abs_x is None:
                running_sum_abs_x = share_x.detach().sum(dim=0)          # [72]
                running_sum_abs_los = share_los.detach().sum(dim=0)      # scalar tensor
                running_sum_signed_x = imp_signed_x.detach().sum(dim=0)  # [72]
                running_sum_signed_los = imp_signed_los.detach().sum(dim=0)  # scalar tensor
                running_sum_delta = delta.detach().sum(dim=0)            # scalar tensor
            else:
                running_sum_abs_x += share_x.detach().sum(dim=0)
                running_sum_abs_los += share_los.detach().sum(dim=0)
                running_sum_signed_x += imp_signed_x.detach().sum(dim=0)
                running_sum_signed_los += imp_signed_los.detach().sum(dim=0)
                running_sum_delta += delta.detach().sum(dim=0)
        else:
            # keep_all path stores per-sample shares for ABS, raw for signed/delta
            scores_abs_x.append(share_x.detach().cpu())
            scores_abs_los.append(share_los.detach().cpu())
            scores_signed_x.append(imp_signed_x.detach().cpu())
            scores_signed_los.append(imp_signed_los.detach().cpu())
            scores_delta.append(delta.detach().cpu())

    # ---- aggregate ----
    if n_seen == 0:
        raise RuntimeError("No samples processed.")

    # fast path: mean without keep_all
    if reduce == "mean" and not keep_all:
        if running_sum_abs_x is None:
            raise RuntimeError("No samples processed (running_sum_abs is None).")
        if running_sum_signed_x is None:
            raise RuntimeError("No samples processed (running_sum_signed is None).")

        # ABS outputs are now mean shares
        global_abs_x = (running_sum_abs_x / float(n_seen)).cpu()          # [72], mean share
        global_abs_los = (running_sum_abs_los / float(n_seen)).cpu()      # scalar, mean share

        # signed + delta unchanged
        global_signed_x = (running_sum_signed_x / float(n_seen)).cpu()
        global_signed_los = (running_sum_signed_los / float(n_seen)).cpu()
        global_delta = (running_sum_delta / float(n_seen)).cpu()

        return global_abs_x, global_abs_los, global_signed_x, global_signed_los, global_delta

    # otherwise, concatenate and reduce
    if len(scores_abs_x) == 0:
        raise RuntimeError("No scores collected.")

    all_abs_x = torch.cat(scores_abs_x, dim=0)        # [n_samples, 72] (shares)
    all_abs_los = torch.cat(scores_abs_los, dim=0)    # [n_samples]      (shares)
    all_signed_x = torch.cat(scores_signed_x, dim=0)  # [n_samples, 72]
    all_signed_los = torch.cat(scores_signed_los, dim=0)  # [n_samples]
    all_delta = torch.cat(scores_delta, dim=0)        # [n_samples]

    if reduce == "mean":
        global_abs_x = all_abs_x.mean(dim=0)          # mean share per feature
        global_abs_los = all_abs_los.mean(dim=0)      # mean share for LOS
        global_signed_x = all_signed_x.mean(dim=0)
        global_signed_los = all_signed_los.mean(dim=0)
        global_delta = all_delta.mean(dim=0)
    elif reduce == "median":
        global_abs_x = all_abs_x.median(dim=0).values
        global_abs_los = all_abs_los.median(dim=0).values
        global_signed_x = all_signed_x.median(dim=0).values
        global_signed_los = all_signed_los.median(dim=0).values
        global_delta = all_delta.median(dim=0).values
    else:
        raise ValueError(reduce)

    return global_abs_x, global_abs_los, global_signed_x, global_signed_los, global_delta # abs: share (ratio) 

from src.utils.seed_set import set_seed
from src.explainers.stablity_report import (
    importance_mean_std_table,
    report
)

def ig_main(
    dataset,
    dataloader,
    model,
    save_path: str,
    edge_index: torch.Tensor,     # [2, E]
    target: str = "logit",        # "logit" only for now
    n_steps: int = 50,
    reduce: Literal["mean", "median"] = "mean",
    keep_all: bool = False,
    max_batches: Optional[int] = None,
    verbose: bool = True,
    sample_ratio: float = 0.1,
    ):
    
    outs_abs_x = []
    outs_abs_los = []
    outs_signed_x = []
    outs_signed_los = []
    outs_delta = []
    
    for s in [0, 1, 2]:
        set_seed(s)
        global_abs_x, global_abs_los, global_signed_x, global_signed_los, global_delta = explain_ig_for_dataset(
            dataloader=dataloader,
            model=model,
            edge_index=edge_index,
            target=target,
            n_steps=n_steps,
            reduce=reduce,
            keep_all=keep_all,
            max_batches=max_batches,
            verbose=verbose,
            sample_ratio=sample_ratio,
            seed=s
        )
        
        outs_abs_x.append(global_abs_x.cpu().float())
        outs_abs_los.append(global_abs_los.cpu().float())
        outs_signed_x.append(global_signed_x.cpu().float())
        outs_signed_los.append(global_signed_los.cpu().float())
        outs_delta.append(global_delta.cpu().float())

    col_names, col_dims, ad_col_index, dis_col_index = dataset.col_info

    df_abs_x = importance_mean_std_table(outs_abs_x, col_names)
    df_abs_los = importance_mean_std_table(outs_abs_los, ["LOS"])
    df_signed_x = importance_mean_std_table(outs_signed_x, col_names)
    df_signed_los = importance_mean_std_table(outs_signed_los, ["LOS"])
    df_delta = importance_mean_std_table(outs_delta, ["delta"])

    report(df_abs_x, outs_abs_x, col_names, save_path, f"IG_abs_x_global_importance.csv")
    report(df_abs_los, outs_abs_los, ["LOS"], save_path, f"IG_abs_los_global_importance.csv", scalar=True)
    report(df_signed_x, outs_signed_x, col_names, save_path, f"IG_signed_x_global_importance.csv")
    report(df_signed_los, outs_signed_los, ["LOS"], save_path, f"IG_signed_los_global_importance.csv", scalar=True)
    report(df_delta, outs_delta, ["delta"], save_path, f"IG_delta_global_importance.csv", scalar=True)