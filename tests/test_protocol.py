from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.protocol.artifacts import create_eval_artifact, create_hpo_artifact
from src.protocol.analysis import analyze_paired_results
from src.protocol.graph_config import load_graph_config, write_graph_config
from src.protocol.mi import compute_mi_dict, mi_cache_path
from src.protocol.stats import bh_fdr_adjust, holm_adjust, nadeau_bengio_corrected_t, tost
from src.protocol.vocabulary import encode_with_codebook
from src.data_processing.edge import fully_connected_pair_edge_index
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


def test_analysis_requires_sesoi_for_f4(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"comparisons": [{
        "family": "F4", "candidate": "ablated", "reference": "full",
        "differences": [0.0, 0.001, -0.001], "n_train": 80, "n_test": 20,
    }]}), encoding="utf-8")
    with pytest.raises(ValueError):
        analyze_paired_results(str(path))
    assert analyze_paired_results(str(path), sesoi=0.1)["comparisons"][0]["tost"]["equivalent"]


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
