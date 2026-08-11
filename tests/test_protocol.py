from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.protocol.artifacts import create_eval_artifact, create_hpo_artifact
from src.protocol.ablations import apply_variant
from src.protocol.analysis import analyze_paired_results, build_paired_results
from src.protocol.graph_config import load_graph_config, write_graph_config
from src.protocol.hpo import apply_trial_params, suggest_protocol_params
from src.protocol.mi import compute_mi_dict, mi_cache_path
from src.protocol.runner import (
    _namespaced_study_name,
    _parse_eval_seeds,
    _redact_storage,
    _require_protocol_codebook,
    _resolve_optuna_storage,
    _select_eval_splits,
    _storage_backend,
    _variant_cfg,
)
from src.protocol.stats import bh_fdr_adjust, holm_adjust, nadeau_bengio_corrected_t, tost
from src.protocol.vocabulary import encode_with_codebook
from src.data_processing.edge import fully_connected_edge_index, fully_connected_pair_edge_index
from src.models.a3tgcn.a3tgcn_2_points import A3TGCN_2_points
from src.models.ctmp_gin.model import CTMPGIN


def test_protocol_artifacts_are_reproducible_and_disjoint(tmp_path):
    labels = np.array([0, 1] * 100)
    eval_artifact = create_eval_artifact(labels, tmp_path / "eval.json")
    hpo_idx = np.asarray(eval_artifact["d_hpo_idx"])
    hpo_artifact = create_hpo_artifact(labels[hpo_idx], tmp_path / "hpo.json", base_indices=hpo_idx)
    assert len(eval_artifact["splits"]) == 15
    assert len(hpo_artifact["subset_folds"]) == 3
    assert len(hpo_artifact["full_folds"]) == 3
    assert set(eval_artifact["d_hpo_idx"]).isdisjoint(eval_artifact["d_eval_idx"])


def test_graph_config_requires_pilot_match(tmp_path):
    pilot = {"artifact_fingerprint": "pilot123"}
    write_graph_config(
        str(tmp_path / "graph_config.json"),
        {"score_method": "raw_mi", "threshold": 0.01, "top_k": 6, "pruning_ratio": 0.3},
        pilot,
    )
    assert load_graph_config(str(tmp_path / "graph_config.json"), pilot)["top_k"] == 6
    with pytest.raises(ValueError):
        load_graph_config(str(tmp_path / "graph_config.json"), {"artifact_fingerprint": "other"})


def test_mi_cache_separates_score_method_but_not_remove_los(tmp_path):
    frame = pd.DataFrame({"A": [0, 0, 1, 1], "B": [0, 1, 0, 1]})
    assert mi_cache_path(str(tmp_path), frame, score_method="raw_mi") != mi_cache_path(str(tmp_path), frame, score_method="nmi")
    assert compute_mi_dict(frame, "nmi")["A"]["B"] >= 0


def test_codebook_oov_and_statistics():
    encoded, oov = encode_with_codebook(pd.DataFrame({"A": [1, 9]}), {"A": [1]})
    assert encoded["A"].tolist() == [0, 1]
    assert oov["A"] == 1
    assert len(holm_adjust([0.01, 0.04])) == 2
    assert len(bh_fdr_adjust([0.01, 0.04])) == 2
    assert nadeau_bengio_corrected_t([0.1, 0.2, 0.15], n_train=80, n_test=20)["n"] == 3
    assert tost([0.0, 0.001, -0.001], sesoi=0.1)["equivalent"]


def test_protocol_runner_requires_codebook_for_data_stages(tmp_path):
    with pytest.raises(SystemExit, match="--codebook is required"):
        _require_protocol_codebook({}, "evaluate")
    codebook = tmp_path / "codebook.json"
    codebook.write_text("{}", encoding="utf-8")
    cfg = {"data": {"codebook_path": str(codebook)}}
    _require_protocol_codebook(cfg, "hpo")
    assert cfg["codebook_path"] == str(codebook)


def test_protocol_optuna_storage_requires_explicit_parallel_safe_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("PROTOCOL_OPTUNA_STORAGE", raising=False)
    monkeypatch.delenv("OPTUNA_STORAGE", raising=False)
    with pytest.raises(SystemExit, match="Optuna stages require --storage"):
        _resolve_optuna_storage(tmp_path)
    sqlite_storage = _resolve_optuna_storage(tmp_path, allow_sqlite_storage=True)
    assert sqlite_storage.startswith("sqlite:///")
    with pytest.raises(SystemExit, match="SQLite Optuna storage is disabled"):
        _resolve_optuna_storage(tmp_path, sqlite_storage)
    postgres = "postgresql+psycopg2://optuna:secret@127.0.0.1:5432/optuna_db"
    assert _resolve_optuna_storage(tmp_path, postgres) == postgres
    assert _storage_backend(postgres) == "postgresql"
    assert "secret" not in _redact_storage(postgres)


def test_protocol_optuna_storage_can_come_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTOCOL_OPTUNA_STORAGE", "postgresql://user:pass@db/optuna")
    assert _resolve_optuna_storage(tmp_path) == "postgresql://user:pass@db/optuna"


def test_protocol_study_names_are_run_dir_namespaced(tmp_path):
    first = _namespaced_study_name(tmp_path / "run_a", "ctmp_gin")
    second = _namespaced_study_name(tmp_path / "run_b", "ctmp_gin")
    assert first != second
    assert first.endswith("__ctmp_gin")
    assert _namespaced_study_name(tmp_path / "run", "ctmp_gin", "paper1") == "paper1__ctmp_gin"


def test_variant_source_validation_blocks_wrong_base_model():
    with pytest.raises(SystemExit, match="C1 must be run with a gin config"):
        _variant_cfg({"model": {"name": "ctmp_gin", "params": {}}}, "C1")
    c1 = _variant_cfg({"model": {"name": "gin", "params": {}}}, "C1")
    assert c1["admission_only"] is True
    with pytest.raises(SystemExit, match="xgboost_admission must be run with a xgboost config"):
        _variant_cfg({"model": {"name": "gin", "params": {}}}, "xgboost_admission")


def test_eval_seed_filter_selects_two_seed_ablation_plan(tmp_path):
    artifact = create_eval_artifact(np.array([0, 1] * 100), tmp_path / "eval.json")
    selected = _select_eval_splits(artifact, _parse_eval_seeds("1,2"))
    assert len(selected) == 10
    assert {split["eval_seed"] for split in selected} == {1, 2}
    with pytest.raises(ValueError, match="requested eval seeds"):
        _select_eval_splits(artifact, (999,))


def test_analysis_requires_sesoi_for_f4(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"comparisons": [{
        "family": "F4", "candidate": "ablated", "reference": "full",
        "differences": [0.0, 0.001, -0.001], "n_train": 80, "n_test": 20,
    }]}), encoding="utf-8")
    with pytest.raises(ValueError):
        analyze_paired_results(str(path))
    assert analyze_paired_results(str(path), sesoi=0.1)["comparisons"][0]["tost"]["equivalent"]


def test_analysis_enforces_single_primary_family(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"comparisons": [
        {"family": "F1", "candidate": "ctmp_gin", "reference": "a3tgcn", "differences": [0.1, 0.2], "n_train": 80, "n_test": 20},
        {"family": "F1", "candidate": "ctmp_gin", "reference": "gin", "differences": [0.1, 0.2], "n_train": 80, "n_test": 20},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="F1 allows at most 1"):
        analyze_paired_results(str(path))


def test_build_paired_results_from_evaluation_summaries(tmp_path):
    split_artifact = create_eval_artifact(np.array([0, 1] * 100), tmp_path / "eval.json")
    split_ids = [split["split_id"] for split in split_artifact["splits"][:3]]
    candidate = {
        "results": [
            {"split_id": split_id, "result": {"test_auc": 0.90 + index * 0.01}}
            for index, split_id in enumerate(split_ids)
        ]
    }
    reference = {
        "results": [
            {"split_id": split_id, "result": {"test_auc": 0.80 + index * 0.01}}
            for index, split_id in enumerate(split_ids)
        ]
    }
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "reference.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    paired = build_paired_results(
        [f"F1,ctmp_gin,a3tgcn,{candidate_path},{reference_path}"],
        tmp_path / "eval.json",
    )
    comparison = paired["comparisons"][0]
    assert comparison["split_ids"] == split_ids
    assert comparison["differences"] == pytest.approx([0.1, 0.1, 0.1])
    assert comparison["n_train"] > comparison["n_test"]


def test_los_as_node_ctmp_path_uses_zero_edge_attributes():
    model = CTMPGIN(
        col_info=(["A", "LOS", "A_D"], [3, 38, 3], [0, 1], [2, 1]),
        embedding_dim=4, gin_hidden_channel=8, gin_1_layers=1,
        gin_hidden_channel_2=8, gin_2_layers=1, num_classes=2,
        dropout_p=0.0, los_embedding_dim=4, train_eps=True,
        readout_mode="last", remove_los_edge=True,
    ).eval()
    x = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge = fully_connected_pair_edge_index(2)
    with torch.no_grad():
        output = model(x, torch.tensor([1, 2]), edge)
    assert output.shape == (2, 1)


def test_ctmp_ct_edge_none_keeps_edge_attr_aligned():
    model = CTMPGIN(
        col_info=(["A", "B", "A_D", "B_D"], [3, 3, 3, 3], [0, 1], [2, 3]),
        embedding_dim=4, gin_hidden_channel=8, gin_1_layers=1,
        gin_hidden_channel_2=8, gin_2_layers=1, num_classes=2,
        dropout_p=0.0, los_embedding_dim=4, train_eps=True,
        readout_mode="last", ct_edge_mode="none",
    ).eval()
    edge = fully_connected_pair_edge_index(2)
    los = torch.tensor([1, 2], dtype=torch.long)
    edge_2, edge_attr = model.get_new_edge(edge, los, batch_size=2)
    assert edge_2.size(1) == edge.size(1)
    assert edge_attr.shape == (2, edge.size(1), 4)
    x = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    with torch.no_grad():
        assert model(x, los, edge).shape == (2, 1)


def test_ctmp_bidirectional_edge_adds_two_cross_directions_and_b3_mask():
    model = CTMPGIN(
        col_info=(["A", "B", "A_D", "B_D"], [3, 3, 3, 3], [0, 1], [2, 3]),
        embedding_dim=4, gin_hidden_channel=8, gin_1_layers=1,
        gin_hidden_channel_2=8, gin_2_layers=1, num_classes=2,
        dropout_p=0.0, los_embedding_dim=4, train_eps=True,
        readout_mode="last", ct_edge_mode="bidirectional",
        fusion_stream_mask=["ad", "dis"],
    ).eval()
    edge = fully_connected_pair_edge_index(2)
    edge_2, edge_attr = model.get_new_edge(edge, torch.tensor([1, 2]), batch_size=2)
    assert edge_2.size(1) == edge.size(1) + 4
    assert edge_attr.shape == (2, edge.size(1) + 4, 4)
    assert model.gated_fusion.num_streams == 2


def test_a3tgcn_protocol_params_are_consumed_by_model():
    model = A3TGCN_2_points(
        batch_size=2,
        col_info=(["A", "B", "A_D", "B_D", "LOS"], [3, 3, 3, 3, 38], [0, 1], [2, 3]),
        embedding_dim=4,
        hidden_channel=8,
        num_classes=2,
        device=torch.device("cpu"),
        num_layers=3,
        dropout_p=0.25,
    )
    assert model.a3tgcn_layer.num_layers == 3
    assert len(model.a3tgcn_layer._base_tgcn_layers) == 3
    assert any(isinstance(module, torch.nn.Dropout) and module.p == 0.25 for module in model.classifier)
    x = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    los = torch.tensor([1, 2], dtype=torch.long)
    with torch.no_grad():
        assert model(x, los, fully_connected_edge_index(2), torch.device("cpu")).shape == (2, 1)


def test_xgboost_hpo_space_excludes_neural_common_params():
    optuna = pytest.importorskip("optuna")
    trial = optuna.create_study(direction="maximize").ask()
    cfg = {"model": {"name": "xgboost", "params": {}}, "train": {}, "edge": {}}
    out = suggest_protocol_params(trial, cfg)
    assert "embedding_dim" not in trial.params
    assert "dropout_p" not in trial.params
    assert "batch_size" not in trial.params
    assert "optimizer" not in out["train"]


def test_apply_trial_params_preserves_fc_edge_ablation_and_a4_metadata():
    cfg = {
        "model": {"name": "ctmp_gin", "params": {}},
        "train": {},
        "edge": {"is_mi_based": False},
    }
    out = apply_trial_params(
        cfg,
        {"batch_size": 256, "learning_rate": 1e-3},
        {"score_method": "nmi", "threshold": 0.05, "top_k": 6, "pruning_ratio": 0.3},
    )
    assert out["edge"]["is_mi_based"] is False
    assert "score_method" not in out["edge"]
    a4 = apply_variant({"model": {"params": {}}}, "A4")
    assert a4["evaluation"]["los_shuffle_repetitions"] == 5
    assert a4["evaluation"]["use_los_shuffle_as_primary"] is True
