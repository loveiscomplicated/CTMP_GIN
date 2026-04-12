# CTMP-GIN
from __future__ import annotations

import os
import sys
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv

cur_dir = os.path.dirname(__file__)
par_dir = os.path.join(cur_dir, '..')
sys.path.append(par_dir)

from src.models.entity_embedding import EntityEmbeddingBatch3
from src.utils.device_set import device_set


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_gin_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Standard 2-layer MLP used inside GINConv / GINEConv."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )


def seperate_x(x: torch.Tensor, ad_idx_t: torch.Tensor, dis_idx_t: torch.Tensor) -> torch.Tensor:
    """Split node features into admission / discharge halves and stack batch-wise.

    Args:
        x: [B, num_vars, emb_dim]
    Returns:
        [B*2, num_nodes, emb_dim]  — first B slices are admission, next B are discharge
    """
    ad_tensor  = torch.index_select(x, dim=1, index=ad_idx_t)   # [B, N, F]
    dis_tensor = torch.index_select(x, dim=1, index=dis_idx_t)  # [B, N, F]
    return torch.cat([ad_tensor, dis_tensor], dim=0)             # [B*2, N, F]


# ---------------------------------------------------------------------------
# GatedFusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x      = torch.cat([A, B, C], dim=-1)   # [B, 3*d]
        logits = self.score(x)                  # [B, 3]
        w      = F.softmax(logits, dim=-1)       # [B, 3]
        A = self.dropout(A)
        B = self.dropout(B)
        C = self.dropout(C)
        fused = w[:, 0:1] * A + w[:, 1:2] * B + w[:, 2:3] * C
        return fused, w, logits


# ---------------------------------------------------------------------------
# CTMPGIN
# ---------------------------------------------------------------------------

ReadoutMode = Literal["concat", "sum", "last"]


class CTMPGIN(nn.Module):
    """Cross-Temporal Message Passing GIN.

    Args:
        readout_mode: How to aggregate node embeddings across GIN layers.
            - ``"concat"``: Concatenate readouts from all layers (GIN paper).
              Projection input dim = ``gin_hidden_channel * gin_1_layers``.
            - ``"sum"``:    Sum readouts across layers. Projection dim unchanged.
            - ``"last"``:   Use only the final layer readout (pre-fix behaviour).
    """

    def __init__(
        self,
        col_info,
        embedding_dim: int,
        gin_hidden_channel: int,
        gin_1_layers: int,
        gin_hidden_channel_2: int,
        gin_2_layers: int,
        num_classes: int,
        dropout_p: float = 0.2,
        los_embedding_dim: int = 8,
        max_los: int = 37,
        train_eps: bool = True,
        gate_hidden_ch: Optional[int] = None,
        remove_proj_ad_dis: bool = False,
        remove_all_proj: bool = False,
        remove_gated_fusion: bool = False,
        readout_mode: ReadoutMode = "concat",
        **kwargs,
    ):
        super().__init__()
        self.device = device_set(kwargs.get("device", "cpu"))
        self.dropout_p = dropout_p
        self.gin_hidden_channel   = gin_hidden_channel
        self.gin_hidden_channel_2 = gin_hidden_channel_2
        self.gin_1_layers = gin_1_layers
        self.gin_2_layers = gin_2_layers

        assert readout_mode in ("concat", "sum", "last"), \
            f"readout_mode must be 'concat', 'sum', or 'last', got {readout_mode!r}"
        self.readout_mode = readout_mode

        self.col_list, self.col_dims, self.ad_col_index, self.dis_col_index = col_info
        self.register_buffer("ad_idx_t",  torch.tensor(self.ad_col_index,  dtype=torch.long))
        self.register_buffer("dis_idx_t", torch.tensor(self.dis_col_index, dtype=torch.long))

        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, embedding_dim=embedding_dim)
        self.embed_los = EntityEmbeddingBatch3(col_dims=[max_los + 1], embedding_dim=los_embedding_dim)

        # GIN stacks
        self.gin_1 = self._build_gin_stack(
            GINConv, embedding_dim, gin_hidden_channel, gin_1_layers, train_eps,
        )
        self.gin_2 = self._build_gin_stack(
            GINEConv, gin_hidden_channel, gin_hidden_channel_2, gin_2_layers, train_eps,
            edge_dim=los_embedding_dim,
        )

        # Projection input dimensions depend on readout_mode
        readout_dim_1 = gin_hidden_channel   * (gin_1_layers if readout_mode == "concat" else 1)
        readout_dim_2 = gin_hidden_channel_2 * (gin_2_layers if readout_mode == "concat" else 1)

        d = gin_hidden_channel
        self.fuse_dim = d

        # Ablation: remove_proj flags require Identity() to be dimension-safe
        if (remove_all_proj or remove_proj_ad_dis) and readout_mode == "concat":
            assert gin_1_layers == 1 and gin_2_layers == 1, (
                "remove_proj_ad_dis / remove_all_proj requires gin_1_layers=1 and "
                "gin_2_layers=1 when readout_mode='concat' (Identity cannot change dims). "
                "Use readout_mode='sum' or 'last' for ablation with multiple layers."
            )

        if remove_all_proj:
            print("proj_ad_dis_merged removed...")
            self.proj_ad     = nn.Identity()
            self.proj_dis    = nn.Identity()
            self.proj_merged = nn.Identity()
        elif remove_proj_ad_dis:
            print("proj_ad_dis removed...")
            self.proj_ad     = nn.Identity()
            self.proj_dis    = nn.Identity()
            self.proj_merged = nn.Linear(readout_dim_2, d)
        else:
            self.proj_ad     = nn.Linear(readout_dim_1, d)
            self.proj_dis    = nn.Linear(readout_dim_1, d)
            self.proj_merged = nn.Linear(readout_dim_2, d)

        self.remove_gated_fusion = remove_gated_fusion
        if not remove_gated_fusion:
            self.gated_fusion = GatedFusion(
                in_dim=3 * self.fuse_dim,
                out_dim=self.fuse_dim,
                hidden_dim=gate_hidden_ch,
                dropout=dropout_p,
            )
        else:
            print("gated_fusion removed...")
            self.gated_fusion = None

        self.classifier_b = nn.Sequential(
            nn.Linear(self.fuse_dim, self.fuse_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.fuse_dim * 2, 1),
        )

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_gin_stack(
        self, conv_cls, in_dim: int, hidden_dim: int, num_layers: int, train_eps: bool, **conv_kwargs
    ) -> nn.ModuleList:
        """Build a stack of GINConv / GINEConv layers."""
        layers = nn.ModuleList()
        layers.append(conv_cls(_make_gin_mlp(in_dim, hidden_dim), train_eps=train_eps, **conv_kwargs))
        for _ in range(num_layers - 1):
            layers.append(conv_cls(_make_gin_mlp(hidden_dim, hidden_dim), train_eps=train_eps, **conv_kwargs))
        return layers

    # ------------------------------------------------------------------
    # Core GIN runner with hierarchical readout
    # ------------------------------------------------------------------

    def _run_gin(
        self,
        layers: nn.ModuleList,
        x_flat: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_ch: int,
        batch_factor: int,
        num_nodes: int,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a GIN stack and return (final node emb, hierarchical graph readout).

        Returns:
            h:         [batch_factor * num_nodes, hidden_ch]  — final node embeddings
            graph_emb: readout shape depends on ``self.readout_mode``:
                       "concat" → [batch_factor, hidden_ch * num_layers]
                       "sum"    → [batch_factor, hidden_ch]
                       "last"   → [batch_factor, hidden_ch]
        """
        h = x_flat
        readouts: list[torch.Tensor] = []
        for layer in layers:
            if edge_attr is None:
                h = layer(x=h, edge_index=edge_index)
            else:
                h = layer(x=h, edge_index=edge_index, edge_attr=edge_attr)
            readouts.append(h.reshape(batch_factor, num_nodes, hidden_ch).mean(dim=1))

        if self.readout_mode == "concat":
            graph_emb = torch.cat(readouts, dim=-1)        # [B_f, F * L]
        elif self.readout_mode == "sum":
            graph_emb = torch.stack(readouts, dim=0).sum(0)  # [B_f, F]
        else:  # "last"
            graph_emb = readouts[-1]                       # [B_f, F]

        return h, graph_emb

    # ------------------------------------------------------------------
    # Shared forward backbone
    # ------------------------------------------------------------------

    def _backbone(
        self,
        x_flat: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int,
        num_nodes: int,
        los: Optional[torch.Tensor] = None,
        edge_index_2: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_internals: bool = False,
    ):
        # --- GIN_1: intra-graph ---
        x_node, intra_readout = self._run_gin(
            self.gin_1, x_flat, edge_index,
            self.gin_hidden_channel, batch_size * 2, num_nodes,
        )
        ad_readout  = intra_readout[:batch_size]   # [B, readout_dim_1]
        dis_readout = intra_readout[batch_size:]   # [B, readout_dim_1]

        # --- GIN_2: inter-graph ---
        if edge_index_2 is None:
            edge_index_2, edge_attr = self.get_new_edge(edge_index, los, batch_size)
        _, inter_readout = self._run_gin(
            self.gin_2, x_node, edge_index_2,
            self.gin_hidden_channel_2, batch_size * 2, num_nodes,
            edge_attr=edge_attr,
        )
        merged_readout = 0.5 * (inter_readout[:batch_size] + inter_readout[batch_size:])  # [B, readout_dim_2]

        # --- Projection ---
        ad_f     = self.proj_ad(ad_readout)          # [B, d]
        dis_f    = self.proj_dis(dis_readout)        # [B, d]
        merged_f = self.proj_merged(merged_readout)  # [B, d]

        # --- Fusion ---
        if self.remove_gated_fusion:
            fused = (ad_f + dis_f + merged_f) / 3.0
            w = logits_gate = None
        else:
            fused, w, logits_gate = self.gated_fusion(ad_f, dis_f, merged_f)

        logit = self.classifier_b(fused)

        if return_internals:
            return logit, fused, w, ad_f, dis_f, merged_f, logits_gate
        return logit

    # ------------------------------------------------------------------
    # Public forward methods
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        los: torch.Tensor,
        edge_index: torch.Tensor,
        device=None,
        return_internals: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        num_nodes  = len(self.ad_idx_t)

        x_embedded = self.entity_embedding_layer(x)
        x_sep  = seperate_x(x_embedded, self.ad_idx_t, self.dis_idx_t)
        x_flat = x_sep.reshape(batch_size * 2 * num_nodes, -1)

        return self._backbone(
            x_flat, edge_index, batch_size, num_nodes,
            los=los, return_internals=return_internals,
        )

    def forward_from_x_emb_with_edge_attr(
        self,
        x_embedded: torch.Tensor,    # [B, num_vars, emb_dim]
        edge_index: torch.Tensor,    # [2, E_internal]
        edge_index_2: torch.Tensor,  # [2, E_total]
        edge_attr: torch.Tensor,     # [E_total, D_los]
    ) -> torch.Tensor:
        """Forward with pre-computed embeddings and pre-computed edge attributes.
        Used by the Integrated Gradients explainer for gradient computation.
        """
        batch_size = x_embedded.size(0)
        num_nodes  = len(self.ad_idx_t)

        x_sep  = seperate_x(x_embedded, self.ad_idx_t, self.dis_idx_t)
        x_flat = x_sep.reshape(batch_size * 2 * num_nodes, -1)

        return self._backbone(
            x_flat, edge_index, batch_size, num_nodes,
            edge_index_2=edge_index_2, edge_attr=edge_attr,
        )

    # ------------------------------------------------------------------
    # Edge utilities (external contracts — do not change signatures)
    # ------------------------------------------------------------------

    def precompute_edge_index_2(self, edge_index: torch.Tensor, batch_size: int) -> None:
        """Cache edge_index_2 once per trial to avoid repeated CPU tensor creation."""
        num_nodes        = len(self.ad_col_index)
        merged_num_nodes = num_nodes * batch_size
        start_node       = torch.arange(0, merged_num_nodes, device=edge_index.device)
        end_node         = start_node + merged_num_nodes
        cross_edge_index = torch.stack([start_node, end_node], dim=0)
        edge_index_2     = torch.cat([edge_index, cross_edge_index], dim=1)
        self.register_buffer("_cached_edge_index_2", edge_index_2)

    def get_new_edge(
        self, edge_index: torch.Tensor, los: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = len(self.ad_col_index)
        if hasattr(self, "_cached_edge_index_2") and self._cached_edge_index_2 is not None:
            edge_index_2 = self._cached_edge_index_2
        else:
            edge_index_2 = self._build_edge_index_2(edge_index, num_nodes, batch_size)
        edge_attr = self.get_edge_attr(los=los, edge_index=edge_index, batch_size=batch_size, num_nodes=num_nodes)
        return edge_index_2, edge_attr

    def _build_edge_index_2(
        self, edge_index: torch.Tensor, num_nodes: int, batch_size: int
    ) -> torch.Tensor:
        merged_num_nodes = num_nodes * batch_size
        start_node       = torch.arange(0, merged_num_nodes, device=edge_index.device)
        end_node         = start_node + merged_num_nodes
        cross_edge_index = torch.stack([start_node, end_node], dim=0)
        return torch.cat([edge_index, cross_edge_index], dim=1)

    def get_edge_index_2(
        self, edge_index: torch.Tensor, num_nodes: int, batch_size: int
    ) -> torch.Tensor:
        return self._build_edge_index_2(edge_index, num_nodes, batch_size)

    def get_edge_attr(
        self,
        los: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int,
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute edge attributes for internal edges (NONE token) and cross edges (LOS token)."""
        device     = edge_index.device
        E_internal = edge_index.size(1)

        # Internal edges → NONE token (index 0), no-copy expand
        none_emb           = self.embed_los.embedding_layer.weight[0]               # (D,)
        edge_attr_internal = none_emb.unsqueeze(0).expand(E_internal, -1)           # (E_internal, D)

        # Cross edges → LOS token, repeated per node
        los_idx        = los.view(batch_size).to(device).long().repeat_interleave(num_nodes)  # (B*N,)
        edge_attr_cross = self.embed_los(los_idx)                                              # (B*N, D)

        return torch.cat([edge_attr_internal, edge_attr_cross], dim=0)  # (E_total, D)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        def _init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(_init)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
