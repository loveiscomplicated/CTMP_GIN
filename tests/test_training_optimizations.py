from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_processing.edge import (
    fully_connected_edge_index,
    fully_connected_edge_index_batched,
    fully_connected_pair_edge_index,
    mi_edge_index_batched,
    mi_edge_index_single,
)
from src.data_processing.mi_dict import _mi_cache_path
from src.data_processing.data_utils import train_test_split_stratified
from src.models.a3tgcn.temporalgcn import TGCN2
from src.models.ctmp_gin.model import CTMPGIN
from src.models.gin.model import GIN
from src.models.gingru.gin_gru_2point import GinGru_2_Point
from scripts.mi_worker import (
    _dataset_kwargs_from_cfg,
    _mi_remove_los_for_cfg as _worker_mi_remove_los_for_cfg,
    _split_seed_from_cfg,
)
from scripts.request_mi import (
    _artifact_key,
    _mi_remove_los_for_cfg as _request_mi_remove_los_for_cfg,
)
import src.data_processing.canonical_teds as canonical_teds


def test_teds_bundle_cache_save_is_best_effort(tmp_path, monkeypatch, capsys) -> None:
    cache_path = tmp_path / "cache.pt"
    original_save = torch.save

    def fail_save(bundle, path):
        original_save({"partial": True}, path)
        raise RuntimeError("zip writer failed")

    monkeypatch.setattr(canonical_teds.torch, "save", fail_save)

    canonical_teds._save_bundle_cache(str(cache_path), object())

    assert not cache_path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert "failed to save dataset cache" in capsys.readouterr().out


def test_mi_top_k_uses_descending_mi_values() -> None:
    mi_dict = {
        "A": pd.Series({"B": 0.1, "C": 0.9}),
        "B": pd.Series({"A": 0.1, "C": 0.8}),
        "C": pd.Series({"A": 0.2, "B": 0.3}),
    }

    edge_index = mi_edge_index_single(mi_dict, top_k=1, threshold=0.0, pruning_ratio=1.0)
    edges = {tuple(edge) for edge in edge_index.t().tolist()}

    assert (0, 2) in edges
    assert (2, 0) in edges
    assert (0, 1) not in edges
    assert (1, 0) not in edges


def test_mi_cache_path_is_split_aware(tmp_path) -> None:
    df = pd.DataFrame({"A": [0, 1, 0, 1], "B": [1, 1, 0, 0], "REASONb": [0, 1, 0, 1]})

    first = _mi_cache_path(str(tmp_path), seed=1, train_df=df.iloc[[0, 1, 2]], remove_los=True)
    second = _mi_cache_path(str(tmp_path), seed=1, train_df=df.iloc[[0, 1, 3]], remove_los=True)

    assert first != second


def test_mi_batched_discharge_edges_start_after_all_admission_graphs() -> None:
    mi_ad_dict = {
        "A": pd.Series({"B": 1.0}),
        "B": pd.Series({"A": 1.0}),
    }
    mi_dis_dict = {
        "A_D": pd.Series({"B_D": 1.0}),
        "B_D": pd.Series({"A_D": 1.0}),
    }

    edge_index = mi_edge_index_batched(
        batch_size=3,
        num_nodes=2,
        mi_ad_dict=mi_ad_dict,
        mi_dis_dict=mi_dis_dict,
        top_k=1,
        threshold=0.0,
        pruning_ratio=1.0,
    )
    edges = {tuple(edge) for edge in edge_index.t().tolist()}

    expected_edges = set()
    for graph_offset in [0, 2, 4, 6, 8, 10]:
        expected_edges.add((graph_offset, graph_offset + 1))
        expected_edges.add((graph_offset + 1, graph_offset))

    assert edges == expected_edges


def test_remote_mi_worker_uses_training_split_and_dataset_options() -> None:
    cfg = {
        "admission_only": True,
        "model": {"name": "gin"},
        "train": {
            "binary": False,
            "ig_label": True,
            "do_preprocess": False,
            "split_seed": 123,
        },
    }

    assert _split_seed_from_cfg(cfg, seed=7) == 123
    assert _split_seed_from_cfg({"train": {}}, seed=7) == 7
    assert _worker_mi_remove_los_for_cfg(cfg) is True
    assert _dataset_kwargs_from_cfg(cfg, remove_los=True) == {
        "binary": False,
        "ig_label": True,
        "remove_los": True,
        "do_preprocess": False,
        "admission_only": True,
    }


def test_remote_mi_artifact_key_tracks_dataset_options() -> None:
    base_cfg = {
        "admission_only": False,
        "model": {"name": "gin"},
        "train": {
            "binary": True,
            "ig_label": False,
            "do_preprocess": True,
            "split_seed": 42,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
    }
    no_preprocess_cfg = {
        **base_cfg,
        "train": {**base_cfg["train"], "do_preprocess": False},
    }
    admission_only_cfg = {**base_cfg, "admission_only": True}

    assert _request_mi_remove_los_for_cfg(base_cfg) is False
    assert _request_mi_remove_los_for_cfg(admission_only_cfg) is True

    first = _artifact_key(
        "single",
        None,
        seed=1,
        cfg=base_cfg,
        remove_los=_request_mi_remove_los_for_cfg(base_cfg),
    )
    second = _artifact_key(
        "single",
        None,
        seed=1,
        cfg=no_preprocess_cfg,
        remove_los=_request_mi_remove_los_for_cfg(no_preprocess_cfg),
    )
    third = _artifact_key(
        "single",
        None,
        seed=1,
        cfg=admission_only_cfg,
        remove_los=_request_mi_remove_los_for_cfg(admission_only_cfg),
    )

    assert first != second
    assert first != third


def test_gin_hidden_layers_do_not_share_mlp_instances() -> None:
    model = GIN(
        embedding_dim=4,
        col_info=(["A", "B", "LOS"], [3, 4, 38], [], []),
        gin_dim=8,
        gin_layer_num=4,
        num_classes=2,
        train_eps=True,
    )

    assert len({id(layer.nn) for layer in model.gin_layers}) == 4


def test_gin_shared_edge_matches_legacy_batched_edge() -> None:
    torch.manual_seed(0)
    batch_size = 3
    num_nodes = 3
    model = GIN(
        embedding_dim=4,
        col_info=(["A", "B", "LOS"], [3, 4, 38], [], []),
        gin_dim=8,
        gin_layer_num=2,
        num_classes=2,
        train_eps=True,
    ).eval()
    x = torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.long)
    los = torch.tensor([1, 5, 9], dtype=torch.long)
    shared_edge = fully_connected_edge_index(num_nodes)
    legacy_edge = torch.cat(
        [shared_edge + batch_idx * num_nodes for batch_idx in range(batch_size)],
        dim=1,
    )

    with torch.no_grad():
        shared_out = model(x, los, shared_edge)
        legacy_out = model(x, los, legacy_edge)

    assert torch.allclose(shared_out, legacy_out, atol=1e-6)


def test_gin_gru_2_point_shared_pair_edge_matches_legacy_batched_edge() -> None:
    torch.manual_seed(1)
    batch_size = 2
    num_nodes = 2
    model = GinGru_2_Point(
        col_info=(["A", "A_D", "LOS"], [3, 3, 38], [0, 2], [1, 2]),
        embedding_dim=4,
        gin_hidden_channel=8,
        train_eps=True,
        gin_layers=2,
        gru_hidden_channel=8,
        num_classes=2,
        dropout_p=0.0,
        gin_layer_out_dropout_p=0.0,
        gru_layer_out_dropout_p=0.0,
    ).eval()
    x = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    los = torch.tensor([1, 7], dtype=torch.long)
    shared_edge = fully_connected_pair_edge_index(num_nodes)
    legacy_edge = fully_connected_edge_index_batched(num_nodes, batch_size=batch_size)

    with torch.no_grad():
        shared_out = model(x, los, shared_edge, device=torch.device("cpu"))
        legacy_out = model(x, los, legacy_edge, device=torch.device("cpu"))

    assert torch.allclose(shared_out, legacy_out, atol=1e-6)


def test_ctmp_gin_shared_pair_edge_matches_legacy_batched_edge() -> None:
    torch.manual_seed(2)
    batch_size = 2
    num_nodes = 2
    model = CTMPGIN(
        col_info=(["A", "B", "A_D", "B_D"], [3, 4, 3, 4], [0, 1], [2, 3]),
        embedding_dim=4,
        gin_hidden_channel=8,
        gin_1_layers=1,
        gin_hidden_channel_2=8,
        gin_2_layers=1,
        num_classes=2,
        dropout_p=0.0,
        los_embedding_dim=4,
        max_los=37,
        train_eps=True,
        readout_mode="last",
    ).eval()
    x = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 2]], dtype=torch.long)
    los = torch.tensor([1, 12], dtype=torch.long)
    shared_edge = fully_connected_pair_edge_index(num_nodes)
    legacy_edge = fully_connected_edge_index_batched(num_nodes, batch_size=batch_size)

    with torch.no_grad():
        shared_out = model(x, los, shared_edge)
        legacy_out = model(x, los, legacy_edge)

    assert torch.allclose(shared_out, legacy_out, atol=1e-6)


def test_tgcn2_fused_gates_match_three_conv_legacy_path() -> None:
    torch.manual_seed(3)
    legacy = TGCN2(4, 5, batch_size=2, cached=False, fuse_gates=False).eval()
    fused = TGCN2(4, 5, batch_size=2, cached=False, fuse_gates=True).eval()

    with torch.no_grad():
        fused.conv_gates.lin.weight.copy_(
            torch.cat(
                [
                    legacy.conv_z.lin.weight,
                    legacy.conv_r.lin.weight,
                    legacy.conv_h.lin.weight,
                ],
                dim=0,
            )
        )
        fused.conv_gates.bias.copy_(
            torch.cat([legacy.conv_z.bias, legacy.conv_r.bias, legacy.conv_h.bias], dim=0)
        )
        fused.linear_z.load_state_dict(legacy.linear_z.state_dict())
        fused.linear_r.load_state_dict(legacy.linear_r.state_dict())
        fused.linear_h.load_state_dict(legacy.linear_h.state_dict())

    x = torch.randn(2, 3, 4)
    edge_index = fully_connected_edge_index(3)

    with torch.no_grad():
        legacy_out = legacy(x, edge_index)
        fused_out = fused(x, edge_index)

    assert torch.allclose(fused_out, legacy_out, atol=1e-6)


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        self.x = torch.arange(20, dtype=torch.long).reshape(10, 2)
        self.y = torch.tensor([0, 1] * 5, dtype=torch.long)
        self.los = torch.arange(1, 11, dtype=torch.long)

    def __len__(self) -> int:
        return self.y.numel()

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.los[index]


def test_loaders_keep_last_batch_when_drop_last_false() -> None:
    train_loader, val_loader, test_loader, _ = train_test_split_stratified(
        _TinyDataset(),
        batch_size=4,
        ratio=[0.6, 0.2, 0.2],
        seed=1,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    assert sum(batch[0].size(0) for batch in train_loader) == len(train_loader.dataset)
    assert sum(batch[0].size(0) for batch in val_loader) == len(val_loader.dataset)
    assert sum(batch[0].size(0) for batch in test_loader) == len(test_loader.dataset)
