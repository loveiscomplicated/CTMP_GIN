# --- drop-in replacement: path loops -> tensorized GPU compute ---
# 핵심 아이디어:
#   path를 "노드열"로 들고 있지 말고, 각 path를 (u_idx, w_idx) 엣지 인덱스 쌍으로 캐시해둔다.
#   그러면 score(path) = sum_e dot( X[w_e]-X[u_e], G[u_e] )
#   를 GPU에서 한 번에 계산 가능.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import torch
import torch.nn as nn
from tqdm import tqdm


BaselineStrategy = Literal["farthest", "fixed"]


# -----------------------------
# Graph utilities (BFS shortest paths)
# -----------------------------

def build_adj_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Build an undirected adjacency list from edge_index [2, E]."""
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    adj = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)
            adj[v].append(u)
    return adj


def bfs_dist_and_parents(adj: List[List[int]], start: int) -> Tuple[List[int], List[List[int]]]:
    """
    BFS from start to compute:
    - dist[v] = shortest distance from start to v (or -1 if unreachable)
    - parents[v] = list of predecessors u such that dist[u] + 1 == dist[v]
    """
    n = len(adj)
    dist = [-1] * n
    parents: List[List[int]] = [[] for _ in range(n)]

    q = [start]
    dist[start] = 0
    head = 0

    while head < len(q):
        u = q[head]
        head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parents[v].append(u)
                q.append(v)
            elif dist[v] == dist[u] + 1:
                parents[v].append(u)

    return dist, parents


def enumerate_shortest_paths(
    parents: List[List[int]],
    start: int,
    target: int,
    max_paths: int = 256,
) -> List[List[int]]:
    """
    Enumerate up to `max_paths` shortest paths from start -> target.
    Returns a list of paths as node lists: [start, ..., target]
    """
    if start == target:
        return [[start]]

    paths: List[List[int]] = []
    stack: List[Tuple[int, List[int]]] = [(target, [target])]

    while stack and len(paths) < max_paths:
        node, suffix = stack.pop()
        if node == start:
            paths.append(list(reversed(suffix)))
            continue
        for p in parents[node]:
            stack.append((p, suffix + [p]))

    return paths


def choose_baseline_node(
    adj: List[List[int]],
    target: int,
    strategy: BaselineStrategy,
    fixed_idx: int = 0,
) -> int:
    """Select baseline node index."""
    n = len(adj)
    if strategy == "fixed":
        if not (0 <= fixed_idx < n):
            raise ValueError(f"fixed_idx {fixed_idx} out of range [0,{n-1}]")
        return fixed_idx

    dist, _ = bfs_dist_and_parents(adj, target)
    max_d = max(dist)
    if max_d <= 0:
        return 0
    candidates = [i for i, d in enumerate(dist) if d == max_d]
    return min(candidates)


# -----------------------------
# NEW: path -> edge-pair tensor cache (u_idx, w_idx)
# -----------------------------

def path_nodes_to_edge_pairs(path: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert node-path [v0, v1, ..., vk] to edge pairs:
      u = [v0, v1, ..., v_{k-1}]
      w = [v1, v2, ..., v_k]
    Returned tensors are on CPU (long).
    """
    if len(path) < 2:
        # no edge
        u = torch.empty((0,), dtype=torch.long)
        w = torch.empty((0,), dtype=torch.long)
        return u, w
    u = torch.tensor(path[:-1], dtype=torch.long)
    w = torch.tensor(path[1:], dtype=torch.long)
    return u, w


class ShortestPathEdgeCache:
    """
    Cache of shortest-path edge pairs for each target node v.

    For a fixed baseline b (per target if baseline_strategy="farthest"),
    we build BFS parents and enumerate shortest paths.
    Then store each path as (u_idx, w_idx) tensors (CPU).

    Access:
      cache.get(v) -> list of (u_idx_cpu, w_idx_cpu)
    """

    def __init__(
        self,
        adj: List[List[int]],
        num_nodes: int,
        baseline_strategy: BaselineStrategy,
        baseline_fixed_idx: int,
        max_paths: int,
    ):
        self.adj = adj
        self.N = num_nodes
        self.baseline_strategy = baseline_strategy
        self.baseline_fixed_idx = baseline_fixed_idx
        self.max_paths = max_paths
        self._cache: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

        # Optional: also store the chosen baseline for each target (debug/analysis)
        self.baseline_of_target: Dict[int, int] = {}

    def build(self):
        """
        Build cache for all targets 0..N-1.
        This is CPU-only and done once.
        """
        for v in range(self.N):
            b = choose_baseline_node(self.adj, v, self.baseline_strategy, self.baseline_fixed_idx)
            self.baseline_of_target[v] = b

            dist, parents = bfs_dist_and_parents(self.adj, b)
            if dist[v] == -1:
                self._cache[v] = []
                continue

            paths = enumerate_shortest_paths(parents, start=b, target=v, max_paths=self.max_paths)
            edge_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for p in paths:
                u_idx, w_idx = path_nodes_to_edge_pairs(p)
                edge_pairs.append((u_idx, w_idx))
            self._cache[v] = edge_pairs

    def get(self, target: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return self._cache.get(target, [])


# -----------------------------
# NEW: tensorized GB-IG for one target node using cached edge pairs
# -----------------------------

def gbig_score_for_node_tensorized(
    X: torch.Tensor,  # [N, F] (GPU)
    G: torch.Tensor,  # [N, F] (GPU)
    edge_pairs_list: List[Tuple[torch.Tensor, torch.Tensor]],  # CPU tensors
    *,
    use_mean: bool = True,
) -> torch.Tensor:
    """
    Tensorized version:
      score(path) = sum_e dot( X[w_e]-X[u_e], G[u_e] )
    computed on GPU via indexing.

    Args:
        X: Node features [N, F] (typically GPU).
        G: Gradients wrt node features [N, F] (same device as X).
        edge_pairs_list: list of (u_idx_cpu, w_idx_cpu) for each shortest path.
        use_mean: if True, average over paths; else sum over paths.

    Returns:
        Scalar tensor on same device as X.
    """
    if len(edge_pairs_list) == 0:
        return X.new_tensor(0.0)

    device = X.device
    path_scores = []

    # Still a loop over paths, but the inner edge loop is removed.
    # Usually max_paths is small (e.g., 16/32/128). Biggest win is eliminating per-edge python loop.
    for (u_cpu, w_cpu) in edge_pairs_list:
        if u_cpu.numel() == 0:
            path_scores.append(X.new_tensor(0.0))
            continue

        u = u_cpu.to(device, non_blocking=True)
        w = w_cpu.to(device, non_blocking=True)

        # [E, F]
        delta = X[w] - X[u]
        gu = G[u]

        # [E] = sum_F delta*gu
        edge_contrib = (delta * gu).sum(dim=-1)

        # scalar
        path_scores.append(edge_contrib.sum())

    scores = torch.stack(path_scores, dim=0)  # [P]
    return scores.mean() if use_mean else scores.sum()


@dataclass
class GBIGResult:
    """Per-batch GB-IG results."""
    gbig_ad: torch.Tensor
    gbig_dis: torch.Tensor
    gbig_var: torch.Tensor


class CTMPGIN_GBIGExplainer:
    """
    GB-IG Explainer for CTMP-GIN with tensorized path scoring.
    (Optimization 3: remove per-edge python loops)
    """

    def __init__(
        self,
        model: nn.Module,
        edge_index_vargraph: torch.Tensor,
        num_nodes: Optional[int] = None,
        ad_indices: Optional[List[int]] = None,
        dis_indices: Optional[List[int]] = None,
        baseline_strategy: BaselineStrategy = "farthest",
        baseline_fixed_idx: int = 0,
        max_paths: int = 128,
        use_abs: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.edge_index = edge_index_vargraph.detach().cpu()
        self.num_nodes = num_nodes
        self.baseline_strategy = baseline_strategy
        self.baseline_fixed_idx = baseline_fixed_idx
        self.max_paths = max_paths
        self.use_abs = use_abs

        if ad_indices is None:
            ad_indices = getattr(model, "ad_col_index", None)
        if dis_indices is None:
            dis_indices = getattr(model, "dis_col_index", None)
        if ad_indices is None or dis_indices is None:
            raise ValueError("Provide ad_indices/dis_indices or ensure model has ad_col_index/dis_col_index.")

        self.ad_indices = list(ad_indices)
        self.dis_indices = list(dis_indices)

        if self.num_nodes is None:
            self.num_nodes = len(self.ad_indices)
        if len(self.ad_indices) != self.num_nodes or len(self.dis_indices) != self.num_nodes:
            raise ValueError("ad_indices and dis_indices must both have length == num_nodes.")

        self.adj = build_adj_list(self.edge_index, self.num_nodes)
        self.device = device

        # NEW: build cache once
        self.path_cache = ShortestPathEdgeCache(
            adj=self.adj,
            num_nodes=self.num_nodes,
            baseline_strategy=self.baseline_strategy,
            baseline_fixed_idx=self.baseline_fixed_idx,
            max_paths=self.max_paths,
        )
        self.path_cache.build()

        # Hook container
        self._captured_emb: Optional[torch.Tensor] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def _register_embedding_hook(self):
        if not hasattr(self.model, "entity_embedding_layer"):
            raise ValueError("Model must have attribute `entity_embedding_layer`.")
        layer = getattr(self.model, "entity_embedding_layer")

        def hook_fn(_module, _inp, out):
            self._captured_emb = out
            out.retain_grad()

        self._hook_handle = layer.register_forward_hook(hook_fn)

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def _infer_device(self, x: torch.Tensor) -> torch.device:
        return self.device if self.device is not None else x.device

    def explain_batch(
        self,
        x: torch.Tensor,
        los: torch.Tensor,
        edge_index: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> GBIGResult:
        device = self._infer_device(x)
        self.model.eval()

        B = x.size(0)
        N = self.num_nodes

        gbig_ad_all = torch.zeros((B, N), device=device)
        gbig_dis_all = torch.zeros((B, N), device=device)

        self._register_embedding_hook()
        try:
            for i in range(B):
                self.model.zero_grad(set_to_none=True)
                self._captured_emb = None

                out = self.model(x, los, edge_index, device=device)

                if out.dim() == 2 and out.size(1) == 1:
                    scalar = out[i, 0]
                elif out.dim() == 2:
                    if target_class is None:
                        raise ValueError("Multi-class output detected; provide target_class.")
                    scalar = out[i, target_class]
                else:
                    raise ValueError(f"Unexpected model output shape: {tuple(out.shape)}")

                scalar.backward(retain_graph=True)

                if self._captured_emb is None or self._captured_emb.grad is None:
                    raise RuntimeError("Failed to capture embedding output/grad via hook.")

                emb_i = self._captured_emb[i]          # [num_vars, D]
                grad_i = self._captured_emb.grad[i]    # [num_vars, D]

                X_ad = emb_i[self.ad_indices]          # [N, D]
                G_ad = grad_i[self.ad_indices]         # [N, D]
                X_dis = emb_i[self.dis_indices]        # [N, D]
                G_dis = grad_i[self.dis_indices]       # [N, D]

                # NEW: per-edge python loop 제거 (path별 텐서 인덱싱)
                for v in range(N):
                    edge_pairs = self.path_cache.get(v)

                    s_ad = gbig_score_for_node_tensorized(
                        X=X_ad, G=G_ad,
                        edge_pairs_list=edge_pairs,
                        use_mean=True,
                    )
                    s_dis = gbig_score_for_node_tensorized(
                        X=X_dis, G=G_dis,
                        edge_pairs_list=edge_pairs,
                        use_mean=True,
                    )

                    gbig_ad_all[i, v] = s_ad
                    gbig_dis_all[i, v] = s_dis

            if self.use_abs:
                gbig_var = 0.5 * (gbig_ad_all.abs() + gbig_dis_all.abs())
            else:
                gbig_var = 0.5 * (gbig_ad_all + gbig_dis_all)

            return GBIGResult(gbig_ad=gbig_ad_all, gbig_dis=gbig_dis_all, gbig_var=gbig_var)

        finally:
            self._remove_hook()


# ---- global importance loop  ----
import itertools
from typing import Optional, List, Literal
from dataclasses import dataclass
import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass
class GlobalImportanceOutput:
    global_importance: torch.Tensor
    all_scores: Optional[torch.Tensor]
    n_samples: int


def _iter_selected_batches(dataloader, selected_indices):
    """
    Yield only selected batches from dataloader, without iterating the full loader.

    Args:
        dataloader: PyTorch DataLoader
        selected_indices: sorted list of batch indices to keep

    Yields:
        (batch_index, batch)
    """
    it = iter(dataloader)
    prev = -1
    for idx in selected_indices:
        skip = idx - prev - 1
        if skip > 0:
            it = itertools.islice(it, skip, None)
        batch = next(it)
        yield idx, batch
        prev = idx


def compute_global_importance_on_loader(
    explainer,  # CTMPGIN_GBIGExplainer
    model: nn.Module,
    dataloader,
    edge_index: torch.Tensor,
    device: torch.device,
    *,
    target_class: Optional[int] = None,
    reduce: Literal["mean", "median"] = "mean",
    keep_all: bool = False,
    max_batches: Optional[int] = None,
    verbose: bool = True,

    # ---- sampling options ----
    sample_ratio: float = 1.0,   # e.g. 0.05 ~ 0.10
    seed: int = 0,
) -> GlobalImportanceOutput:
    """
    Compute dataset-level global importance with optional batch-level sampling.
    tqdm progress reflects *actual* processed batches.
    """

    model.eval()

    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1].")

    scores_cpu: List[torch.Tensor] = []
    running_sum: Optional[torch.Tensor] = None
    n_seen = 0

    # ---- determine total batches ----
    if not hasattr(dataloader, "__len__"):
        raise ValueError("Sampling requires dataloader with __len__().")

    total_batches = len(dataloader)
    effective_batches = min(total_batches, max_batches) if max_batches is not None else total_batches

    # ---- sample batch indices ----
    if sample_ratio < 1.0:
        g = torch.Generator().manual_seed(seed)
        n_pick = max(1, int(round(effective_batches * sample_ratio)))
        perm = torch.randperm(effective_batches, generator=g).tolist()
        selected = sorted(perm[:n_pick])
        if verbose:
            print(f"[GB-IG] Sampling batches: {n_pick}/{effective_batches} (ratio={sample_ratio}, seed={seed})")
    else:
        selected = list(range(effective_batches))

    # ---- iterate ONLY selected batches ----
    iterator = _iter_selected_batches(dataloader, selected)

    pbar = tqdm(iterator, total=len(selected), desc="GB-IG (dataset)")

    processed_batches = 0

    for b_idx, batch in pbar:
        x, y, los = batch
        x = x.to(device)
        los = los.to(device)

        res = explainer.explain_batch(
            x=x,
            los=los,
            edge_index=edge_index,
            target_class=target_class,
        )
        scores_b = res.gbig_var  # [B, N]

        B = scores_b.size(0)
        n_seen += B
        processed_batches += 1

        if reduce == "mean" and not keep_all:
            if running_sum is None:
                running_sum = scores_b.detach().sum(dim=0)
            else:
                running_sum += scores_b.detach().sum(dim=0)
        else:
            scores_cpu.append(scores_b.detach().cpu())

        '''if verbose and processed_batches % 10 == 0:
            print(f"[GB-IG] processed_batches={processed_batches} | samples so far: {n_seen}")'''

        model.zero_grad(set_to_none=True)

    # ---- aggregate ----
    if reduce == "mean" and not keep_all:
        if running_sum is None or n_seen == 0:
            raise RuntimeError("No samples processed.")
        global_importance = (running_sum / float(n_seen)).cpu()
        return GlobalImportanceOutput(
            global_importance=global_importance,
            all_scores=None,
            n_samples=n_seen,
        )

    if len(scores_cpu) == 0:
        raise RuntimeError("No scores collected.")

    all_scores = torch.cat(scores_cpu, dim=0)

    if reduce == "mean":
        global_importance = all_scores.mean(dim=0)
    elif reduce == "median":
        global_importance = all_scores.median(dim=0).values
    else:
        raise ValueError(reduce)

    return GlobalImportanceOutput(
        global_importance=global_importance,
        all_scores=all_scores if keep_all else None,
        n_samples=all_scores.size(0),
    )
