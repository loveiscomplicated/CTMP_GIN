from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.data_processing.edge import fully_connected_edge_index_batched
from src.models.ctmp_gin import (
    HybridOrdinalLOSEncoder,
    ensure_ctmp_gin_los_encoder_defaults,
)
from src.models.ctmp_gin.model import CTMPGIN


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: str) -> dict:
    with (REPO_ROOT / path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_ctmp_model(*, los_emb: str) -> CTMPGIN:
    return CTMPGIN(
        col_info=(
            ["A", "B", "A_D", "B_D"],
            [3, 4, 3, 4],
            [0, 1],
            [2, 3],
        ),
        embedding_dim=6,
        gin_hidden_channel=8,
        gin_1_layers=1,
        gin_hidden_channel_2=8,
        gin_2_layers=1,
        num_classes=2,
        dropout_p=0.0,
        los_embedding_dim=5,
        max_los=37,
        train_eps=True,
        readout_mode="last",
        los_emb=los_emb,
        los_ordinal_basis_dim=6,
        los_ordinal_dim=4,
        los_ordinal_hidden_dim=10,
    )


def test_ctmp_gin_yaml_declares_los_encoder_defaults() -> None:
    cfg = _load_yaml("configs/ctmp_gin.yaml")
    ensure_ctmp_gin_los_encoder_defaults(cfg)
    params = cfg["model"]["params"]

    assert params["los_emb"] == "embedding"
    assert params["los_ordinal_basis_dim"] == 8
    assert params["los_ordinal_dim"] == 8
    assert params["los_ordinal_hidden_dim"] == 32


def test_ensure_ctmp_gin_los_encoder_defaults_is_ctmp_only() -> None:
    ctmp_cfg = {"model": {"name": "ctmp_gin", "params": {}}}
    ensure_ctmp_gin_los_encoder_defaults(ctmp_cfg)
    assert ctmp_cfg["model"]["params"]["los_emb"] == "embedding"

    gin_cfg = {"model": {"name": "gin", "params": {}}}
    ensure_ctmp_gin_los_encoder_defaults(gin_cfg)
    assert "los_emb" not in gin_cfg["model"]["params"]


def test_hybrid_ordinal_los_encoder_shapes() -> None:
    encoder = HybridOrdinalLOSEncoder(num_los_classes=37, out_dim=8)

    assert encoder.all_embeddings().shape == (37, 8)
    assert encoder(torch.tensor([0, 1, 36], dtype=torch.long)).shape == (3, 8)


def test_hybrid_ordinal_los_encoder_backward_reaches_trainable_params() -> None:
    encoder = HybridOrdinalLOSEncoder(num_los_classes=37, out_dim=8)
    out = encoder(torch.tensor([0, 10, 36], dtype=torch.long))
    loss = out.pow(2).mean()
    loss.backward()

    assert encoder.ordinal_increments.grad is not None
    linear_layers = [m for m in encoder.proj if isinstance(m, torch.nn.Linear)]
    assert linear_layers[0].weight.grad is not None
    assert linear_layers[1].weight.grad is not None


def test_ctmp_gin_edge_attr_shape_matches_across_los_encoder_modes() -> None:
    batch_size = 2
    num_nodes = 2
    edge_index = fully_connected_edge_index_batched(num_nodes=num_nodes, batch_size=batch_size)
    los_hard = torch.tensor([1, 37], dtype=torch.long)

    emb_model = _build_ctmp_model(los_emb="embedding")
    hybrid_model = _build_ctmp_model(los_emb="hybrid_ordinal")

    emb_attr_hard = emb_model.get_edge_attr(los_hard, edge_index=edge_index, batch_size=batch_size, num_nodes=num_nodes)
    hybrid_attr_hard = hybrid_model.get_edge_attr(los_hard, edge_index=edge_index, batch_size=batch_size, num_nodes=num_nodes)

    assert emb_attr_hard.shape == hybrid_attr_hard.shape
    assert emb_attr_hard.shape[1] == 5


def test_ctmp_gin_hybrid_ordinal_forward_smoke() -> None:
    batch_size = 2
    num_nodes = 2
    model = _build_ctmp_model(los_emb="hybrid_ordinal")
    x = torch.tensor([[0, 1, 0, 1], [2, 3, 1, 2]], dtype=torch.long)
    los = torch.tensor([1, 12], dtype=torch.long)
    edge_index = fully_connected_edge_index_batched(num_nodes=num_nodes, batch_size=batch_size)

    out = model(x, los, edge_index)
    loss = out.pow(2).mean()
    loss.backward()

    assert out.shape == (batch_size, 1)
    assert model.los_encoder is not None
    assert model.los_encoder.ordinal_increments.grad is not None
