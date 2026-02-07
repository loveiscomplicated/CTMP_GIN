import torch
import torch.nn.functional as F

@torch.no_grad()
def _make_baseline_x_like(x_idx: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x_idx, dtype=torch.long)

@torch.no_grad()
def _make_baseline_los_like(los_idx: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(los_idx, dtype=torch.long)

def manual_ig_embeddings(
    model,
    x_idx: torch.Tensor,          # [B, 72] long
    los_idx: torch.Tensor,        # [B] long
    edge_index: torch.Tensor,     # [2, E]
    target: str = "logit",        # "logit" only for now
    n_steps: int = 50,
):
    """
    Returns:
      ig_x_emb:  [B, 72, emb_dim]  (variable attribution in embedding space)
      ig_los_emb:[B, los_emb_dim]  (LOS attribution in embedding space)
      delta:     [B] (approx convergence check: f(x)-f(baseline) - sum(IG))
    """
    model.eval()
    device = next(model.parameters()).device

    x_idx = x_idx.to(device).long()
    los_idx = los_idx.to(device).long()
    edge_index = edge_index.to(device)

    # baselines (indices)
    base_x_idx = _make_baseline_x_like(x_idx).to(device)
    base_los_idx = _make_baseline_los_like(los_idx).to(device)

    # embeddings (continuous)
    # entity_embedding_layer(x_idx) -> [B, 72, emb_dim]
    x_emb = model.entity_embedding_layer(x_idx)
    base_x_emb = model.entity_embedding_layer(base_x_idx)

    los_emb = model.embed_los(los_idx)          # [B, los_emb_dim] or [B,1,los_emb_dim] depending on impl
    base_los_emb = model.embed_los(base_los_idx)

    # Normalize LOS embed shape to [B, D]
    if los_emb.dim() == 3:
        los_emb = los_emb.squeeze(1)
        base_los_emb = base_los_emb.squeeze(1)

    # We will IG over x_emb and los_emb separately, holding the other at actual value.
    # (필요하면 둘을 동시에 경로로 묶는 것도 가능하지만, "무조건"은 분리 경로가 안전)

    # ----- helper: forward with given x_emb + los_idx (indices) -----
    def f_from_xemb(xemb, los_indices):
        # model uses los_indices (long) internally to build edge_attr, so keep it as indices.
        out = model.forward_from_x_emb(xemb, los_indices, edge_index)
        out = out.squeeze(-1)  # [B]
        return out

    # ----- helper: forward with given los_emb (continuous) -----
    # los는 edge_attr로 들어가는데, 지금 모델은 los_idx로 embed_los를 내부에서 만들고 있음.
    # 그래서 los_emb로 직접 주입하려면 "get_edge_attr"를 los_emb로 받는 버전이 필요.
    # -> 무조건 돌아가게 하려면 LOS는 "embed_los.embedding_layer"에 대한 IG를 따로 말고,
    #    일단은 LOS attribution을 'edge_attr gradient'로 계산하는 방식으로 간다.
    #
    # 즉, LOS는 IG가 아니라: edge_attr에 대한 gradient × (edge_attr - baseline_edge_attr)로 처리.
    # (네 구조에서 이게 가장 단단함)

    # =========================
    # 1) Variable IG (x_emb)
    # =========================
    # allocate
    ig_x = torch.zeros_like(x_emb, device=device)
    total_grads = torch.zeros_like(x_emb, device=device)

    # difference
    dx = (x_emb - base_x_emb)

    # IG loop
    for s in range(1, n_steps + 1):
        alpha = float(s) / n_steps
        x_s = (base_x_emb + alpha * dx).detach()
        x_s.requires_grad_(True)

        out = f_from_xemb(x_s, los_idx)          # [B]
        # sum to scalar
        out_sum = out.sum()

        grads = torch.autograd.grad(out_sum, x_s, retain_graph=False, create_graph=False)[0]
        total_grads += grads

    avg_grads = total_grads / n_steps
    ig_x = dx * avg_grads  # [B, 72, emb_dim]

    # =========================
    # 2) LOS attribution (robust): edge_attr gradient attribution
    # =========================
    # We compute grad w.r.t. edge_attr produced by los_idx inside get_edge_attr.
    # To do so, we re-run forward once, but we need access to edge_attr tensor with requires_grad=True.
    #
    # 가장 간단한 방법: get_new_edge를 살짝 바꿔서 edge_attr를 반환할 때 requires_grad_를 걸어줌.
    # 여기서는 "모델 코드를 거의 안 건드리고" 하기 위해 forward 중간에서 edge_attr를 만들고 직접 gin_2까지 흉내내는 건 복잡.
    #
    # 그래서 "무조건" 기준으로는 CTMPGIN에 아래 메서드 1개만 더 추가하는 걸 권장:
    #   forward_with_edge_attr(x_embedded, edge_index, edge_index_2, edge_attr) -> logit
    #
    # 하지만 일단 지금 당장 돌아가게 하는 최소 수정은:
    #   get_new_edge에서 edge_attr에 requires_grad_를 걸고, forward가 그걸 그대로 쓰게 한 뒤
    #   forward 직후 edge_attr.grad로 attribution 계산

    # ---- compute LOS grad attribution by temporary patch via hook ----
    saved = {}

    def _edge_attr_hook(module, inp, out):
        # not used
        return

    # monkeypatch: wrap original get_new_edge
    orig_get_new_edge = model.get_new_edge

    def patched_get_new_edge(edge_index, los, batch_size):
        ei2, ea = orig_get_new_edge(edge_index=edge_index, los=los, batch_size=batch_size)
        ea = ea.detach()
        ea.requires_grad_(True)
        saved["edge_attr"] = ea
        saved["edge_index_2"] = ei2
        return ei2, ea

    model.get_new_edge = patched_get_new_edge

    # one forward for LOS grad
    x_emb_det = x_emb.detach()
    x_emb_det.requires_grad_(False)

    out = f_from_xemb(x_emb_det, los_idx)  # [B]
    out_sum = out.sum()
    out_sum.backward()

    edge_attr = saved["edge_attr"]              # [E_total, D_los]
    grad_edge_attr = edge_attr.grad             # same shape

    # baseline edge_attr: internal edges are NONE(0), cross edges are los baseline(0) as well
    # so baseline edge_attr == embed_los(0) for all edges
    with torch.no_grad():
        base_edge_attr = model.embed_los(torch.zeros(edge_attr.size(0), device=device, dtype=torch.long))
        if base_edge_attr.dim() == 3:
            base_edge_attr = base_edge_attr.squeeze(1)

    ig_edge_attr = (edge_attr - base_edge_attr) * grad_edge_attr  # [E_total, D_los]

    # restore
    model.get_new_edge = orig_get_new_edge
    model.zero_grad(set_to_none=True)

    # Aggregate LOS contribution only on cross edges (the last B*num_nodes edges in your construction)
    B = x_idx.size(0)
    num_nodes = len(model.ad_col_index)
    E_cross = B * num_nodes
    E_total = ig_edge_attr.size(0)
    cross = ig_edge_attr[E_total - E_cross:]  # [B*N, D]

    # reduce per-sample: reshape [B, N, D] -> sum over N
    cross = cross.reshape(B, num_nodes, -1).sum(dim=1)  # [B, D_los]
    ig_los = cross  # [B, D_los]

    # =========================
    # 3) Convergence-ish delta
    # =========================
    with torch.no_grad():
        fx = f_from_xemb(x_emb, los_idx)  # [B]
        f0 = f_from_xemb(base_x_emb, base_los_idx)  # [B]

        # total attribution scalar per sample
        ig_x_scalar = ig_x.sum(dim=(1, 2))          # [B]
        ig_los_scalar = ig_los.sum(dim=1)           # [B]
        delta = (fx - f0) - (ig_x_scalar + ig_los_scalar)

    return ig_x, ig_los, delta

