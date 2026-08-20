from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.protocol.artifacts import create_eval_artifact, create_hpo_artifact
from src.protocol.ablations import VARIANTS, apply_variant, validate_ablation_mutation
from src.protocol.analysis import analyze_paired_results, build_paired_results
from src.protocol.graph_config import load_graph_config, write_graph_config
from src.protocol.hpo import apply_trial_params, normalize_graph_params, suggest_protocol_params
from src.protocol.mi import compute_mi_dict, mi_cache_path
from src.protocol.runner import (
    finalize_evaluation,
    finalize_top5_reevaluation,
    _namespaced_study_name,
    _optimize_study,
    _parse_eval_split_ids,
    _parse_eval_seeds,
    _redact_storage,
    _ensure_protocol_codebook,
    _require_protocol_codebook,
    _resolve_optuna_storage,
    _select_eval_splits,
    _storage_backend,
    _validate_selected_config_for_variant,
    _variant_cfg,
    prepare_top5_reevaluation,
    run_top5_score,
    top5_pending_scores,
)
from src.protocol.stats import bh_fdr_adjust, holm_adjust, nadeau_bengio_corrected_t, tost
from src.protocol.vocabulary import encode_with_codebook, load_codebook, write_codebook_from_csv
import src.protocol.runner as protocol_runner
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
        model_name="ctmp_gin",
    )
    loaded = load_graph_config(str(tmp_path / "graph_config.json"), pilot, model_name="gin")
    assert loaded["top_k"] == 6
    assert loaded["source_model_name"] == "ctmp_gin"
    assert "gin" in loaded["compatible_model_names"]
    with pytest.raises(ValueError, match="not marked compatible"):
        load_graph_config(str(tmp_path / "graph_config.json"), pilot, model_name="unsupported_model")
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
    nb = nadeau_bengio_corrected_t([0.1, 0.2, 0.15], n_train=80, n_test=20)
    assert nb["n"] == 3
    assert nb["test_train_ratio"] == pytest.approx(0.25)
    assert nadeau_bengio_corrected_t([0.1, 0.2, 0.15], test_train_ratio=0.25)["correction"] == pytest.approx(nb["correction"])
    assert tost([0.0, 0.001, -0.001], sesoi=0.1)["equivalent"]


def test_protocol_runner_requires_codebook_for_data_stages(tmp_path):
    with pytest.raises(SystemExit, match="--codebook is required"):
        _require_protocol_codebook({}, "evaluate")
    codebook = tmp_path / "codebook.json"
    codebook.write_text("{}", encoding="utf-8")
    cfg = {"data": {"codebook_path": str(codebook)}}
    _require_protocol_codebook(cfg, "hpo")
    assert cfg["codebook_path"] == str(codebook)


def test_protocol_auto_generates_codebook_from_preprocessed_source(tmp_path):
    root = tmp_path / "data"
    raw = root / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame({
        "DISYR": [2022, 2022],
        "CASEID": [1, 2],
        "A": [2, 1],
        "LOS": [1, 2],
        "REASON": [1, 2],
    }).to_csv(raw / "TEDS_Discharge.csv", index=False)
    pd.DataFrame({
        "DISYR": [2022, 2022],
        "CASEID": [1, 2],
        "A": [3, 1],
        "LOS": [1, 3],
        "REASON": [1, 2],
    }).to_csv(raw / "missing_corrected.csv", index=False)

    cfg = {"train": {"do_preprocess": True}}
    _ensure_protocol_codebook(cfg, "prepare", str(root), str(tmp_path / "run"))

    codebook_path = tmp_path / "run" / "codebook.json"
    assert cfg["codebook_path"] == str(codebook_path.resolve())
    assert load_codebook(str(codebook_path)) == {"A": [1, 3], "LOS": [1, 3]}


def test_protocol_auto_codebook_runs_preprocessing_when_cache_missing(tmp_path, monkeypatch):
    root = tmp_path / "data"
    raw = root / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame({
        "DISYR": [2022, 2022],
        "CASEID": [1, 2],
        "A": [-9, 1],
        "LOS": [1, 2],
        "REASON": [1, 2],
    }).to_csv(raw / "TEDS_Discharge.csv", index=False)
    calls = []

    def fake_preprocess(raw_path, missing_path):
        calls.append((raw_path, missing_path))
        return pd.DataFrame({
            "DISYR": [2022, 2022],
            "CASEID": [1, 2],
            "A": [0, 1],
            "LOS": [1, 2],
            "REASON": [1, 2],
        })

    monkeypatch.setattr(protocol_runner, "tackle_missing_value_wrapper", fake_preprocess)

    cfg = {"train": {"do_preprocess": True}}
    _ensure_protocol_codebook(cfg, "prepare", str(root), str(tmp_path / "run"))

    assert calls == [(str(raw / "TEDS_Discharge.csv"), str(raw / "missing_corrected.csv"))]
    assert load_codebook(str(tmp_path / "run" / "codebook.json")) == {"A": [0, 1], "LOS": [1, 2]}


def test_protocol_auto_codebook_reuses_existing_file(tmp_path, monkeypatch):
    existing = tmp_path / "run" / "codebook.json"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"A": [1]}', encoding="utf-8")
    monkeypatch.setattr(
        protocol_runner,
        "tackle_missing_value_wrapper",
        lambda *_: pytest.fail("existing auto codebook should be reused"),
    )

    cfg = {"train": {"do_preprocess": True}}
    _ensure_protocol_codebook(cfg, "hpo", str(tmp_path / "data"), str(tmp_path / "run"))

    assert cfg["codebook_path"] == str(existing.resolve())


def test_protocol_auto_codebook_uses_variant_specific_target(tmp_path):
    root = tmp_path / "data"
    raw = root / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame({
        "DISYR": [2022, 2022],
        "CASEID": [1, 2],
        "A": [-9, 1],
        "LOS": [1, 2],
        "REASON": [1, 2],
    }).to_csv(raw / "TEDS_Discharge.csv", index=False)

    cfg = {"train": {"do_preprocess": False}}
    _ensure_protocol_codebook(
        cfg,
        "ablation-hpo",
        str(root),
        str(tmp_path / "run"),
        variant="w/o_preprocessing",
    )

    codebook_path = tmp_path / "run" / "codebooks" / "w_o_preprocessing.json"
    assert cfg["codebook_path"] == str(codebook_path.resolve())
    assert load_codebook(str(codebook_path)) == {"A": [-9, 1], "LOS": [1, 2]}


def test_write_codebook_from_csv_omits_id_and_label_columns(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "DISYR,CASEID,A,REASON\n2022,1,2,1\n2022,2,1,2\n",
        encoding="utf-8",
    )
    report = write_codebook_from_csv(source, tmp_path / "codebook.json")
    assert report["feature_columns"] == 1
    assert load_codebook(str(tmp_path / "codebook.json")) == {"A": [1, 2]}


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


def test_controlled_ablation_variants_inherit_full_selected_config():
    for variant in ["A1", "A2", "A3", "w/o_gated_fusion", "w/o_mi_edge", "w/o_preprocessing"]:
        assert VARIANTS[variant]["hpo"] is False
        _validate_selected_config_for_variant(
            {"variant": "full", "model_name": "ctmp_gin", "params": {}},
            {"model": {"name": "ctmp_gin"}},
            variant,
            "selected.json",
        )


def test_ablation_mutation_diff_is_whitelisted():
    parent = {"model": {"params": {}}, "edge": {"is_mi_based": True}, "train": {"do_preprocess": True}}
    effective = apply_variant(parent, "w/o_mi_edge")
    report = validate_ablation_mutation(parent, effective, "w/o_mi_edge")
    assert report["diffs"] == [{"path": "edge.is_mi_based", "before": True, "after": False}]
    unexpected = json.loads(json.dumps(effective))
    unexpected["train"]["batch_size"] = 512
    with pytest.raises(ValueError, match="non-whitelisted"):
        validate_ablation_mutation(parent, unexpected, "w/o_mi_edge")
    a4_report = validate_ablation_mutation({"model": {"params": {}}}, apply_variant({"model": {"params": {}}}, "A4"), "A4")
    assert {item["path"] for item in a4_report["diffs"]} == {
        "evaluation.los_shuffle_repetitions",
        "evaluation.use_los_shuffle_as_primary",
    }


def test_eval_seed_filter_selects_two_seed_ablation_plan(tmp_path):
    artifact = create_eval_artifact(np.array([0, 1] * 100), tmp_path / "eval.json")
    selected = _select_eval_splits(artifact, _parse_eval_seeds("1,2"))
    assert len(selected) == 10
    assert {split["eval_seed"] for split in selected} == {1, 2}
    with pytest.raises(ValueError, match="requested eval seeds"):
        _select_eval_splits(artifact, (999,))


def test_eval_split_id_filter_selects_exact_splits(tmp_path):
    artifact = create_eval_artifact(np.array([0, 1] * 100), tmp_path / "eval.json")
    split_ids = [artifact["splits"][0]["split_id"], artifact["splits"][3]["split_id"]]
    selected = _select_eval_splits(
        artifact,
        eval_split_ids=_parse_eval_split_ids([",".join(split_ids)]),
    )
    assert [split["split_id"] for split in selected] == split_ids
    with pytest.raises(ValueError, match="requested eval split_ids"):
        _select_eval_splits(artifact, eval_split_ids=("missing_split",))


def test_optimize_study_respects_shared_trial_budget():
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study(direction="maximize")

    def objective(trial):
        return float(trial.number)

    _optimize_study(study, objective, n_trials=5, max_total_trials=2)
    assert len(study.trials) == 2
    _optimize_study(study, objective, n_trials=5, max_total_trials=2)
    assert len(study.trials) == 2


def test_top5_reevaluation_scores_are_resume_safe(tmp_path, monkeypatch):
    optuna = pytest.importorskip("optuna")
    labels = np.array([0, 1] * 200)
    run_dir = tmp_path / "run"
    eval_artifact = create_eval_artifact(labels, run_dir / "d_eval_split_artifact.json")
    hpo_idx = np.asarray(eval_artifact["d_hpo_idx"])
    create_hpo_artifact(labels[hpo_idx], run_dir / "d_hpo_split_artifact.json", base_indices=hpo_idx)

    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    study = optuna.create_study(
        study_name=_namespaced_study_name(run_dir, "gin_protocol"),
        storage=storage,
        direction="maximize",
    )
    values = [0.70, 0.95, 0.90]
    study.optimize(lambda trial: values[trial.number], n_trials=len(values))

    cfg = {"model": {"name": "gin", "params": {}}, "edge": {"is_mi_based": True}, "train": {}}
    manifest = prepare_top5_reevaluation(
        cfg,
        str(run_dir),
        storage=storage,
        allow_sqlite_storage=True,
        top_n=2,
    )

    assert [item["trial_number"] for item in manifest["candidate_trials"]] == [1, 2]
    assert len(top5_pending_scores(run_dir)) == 6

    calls = []

    def fake_score(cfg, root, fold_info, trial_number):
        calls.append((trial_number, int(fold_info["fold"])))
        return 0.80 + trial_number * 0.01 + int(fold_info["fold"]) * 0.001

    monkeypatch.setattr(protocol_runner, "_score_config", fake_score)

    run_top5_score(cfg, str(tmp_path / "data"), str(run_dir), 1, 0)
    assert len(top5_pending_scores(run_dir)) == 5
    run_top5_score(cfg, str(tmp_path / "data"), str(run_dir), 1, 0)
    assert calls == [(1, 0)]

    for item in top5_pending_scores(run_dir):
        run_top5_score(
            cfg,
            str(tmp_path / "data"),
            str(run_dir),
            item["trial_number"],
            item["fold_index"],
        )

    selected = finalize_top5_reevaluation(cfg, str(run_dir))
    assert selected["trial_number"] == 2
    assert top5_pending_scores(run_dir) == []
    assert (run_dir / "selected_config.json").exists()


def test_finalize_evaluation_writes_summary_and_rejects_missing_splits(tmp_path):
    labels = np.array([0, 1] * 100)
    run_dir = tmp_path / "run"
    eval_artifact = create_eval_artifact(labels, run_dir / "d_eval_split_artifact.json")
    hpo_idx = np.asarray(eval_artifact["d_hpo_idx"])
    create_hpo_artifact(labels[hpo_idx], run_dir / "d_hpo_split_artifact.json", base_indices=hpo_idx)
    split_ids = [split["split_id"] for split in eval_artifact["splits"][:2]]
    (run_dir / "evaluation").mkdir(parents=True)
    for index, split_id in enumerate(split_ids):
        (run_dir / "evaluation" / f"{split_id}.json").write_text(
            json.dumps({"split_id": split_id, "result": {"test_auc": 0.8 + index * 0.01}}),
            encoding="utf-8",
        )

    summary = finalize_evaluation(
        {"model": {"name": "ctmp_gin"}},
        str(run_dir),
        eval_split_ids=tuple(split_ids),
    )
    assert summary["count"] == 2
    assert json.loads((run_dir / "evaluation_summary.json").read_text(encoding="utf-8"))["count"] == 2
    with pytest.raises(FileNotFoundError, match="missing evaluation split outputs"):
        finalize_evaluation(
            {"model": {"name": "ctmp_gin"}},
            str(run_dir),
        )


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
        "graph_config_fingerprint": "graph123",
        "results": [
            {"split_id": split_id, "result": {"test_auc": 0.90 + index * 0.01}}
            for index, split_id in enumerate(split_ids)
        ]
    }
    reference = {
        "graph_config_fingerprint": "graph123",
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
    expected_n_train = len(split_artifact["splits"][0]["train_idx"]) + len(split_artifact["splits"][0]["val_idx"])
    expected_n_test = len(split_artifact["splits"][0]["test_idx"])
    assert comparison["n_train"] == expected_n_train
    assert comparison["n_test"] == expected_n_test
    assert comparison["n_train_values"] == [expected_n_train] * 3
    assert comparison["n_test_values"] == [expected_n_test] * 3
    assert comparison["split_sizes_constant"] is True
    assert comparison["nb_test_train_ratio"] == pytest.approx(expected_n_test / expected_n_train)


def test_build_paired_results_rejects_unpaired_splits_but_allows_graph_mismatch(tmp_path):
    split_artifact = create_eval_artifact(np.array([0, 1] * 100), tmp_path / "eval.json")
    split_ids = [split["split_id"] for split in split_artifact["splits"][:2]]
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "reference.json"
    candidate_path.write_text(json.dumps({
        "graph_config_fingerprint": "graph123",
        "results": [{"split_id": split_id, "result": {"test_auc": 0.9}} for split_id in split_ids],
    }), encoding="utf-8")
    reference_path.write_text(json.dumps({
        "graph_config_fingerprint": "graph123",
        "results": [{"split_id": split_ids[0], "result": {"test_auc": 0.8}}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unpaired split sets"):
        build_paired_results(
            [f"F1,ctmp_gin,a3tgcn,{candidate_path},{reference_path}"],
            tmp_path / "eval.json",
        )

    reference_path.write_text(json.dumps({
        "graph_config_fingerprint": "other_graph",
        "results": [{"split_id": split_id, "result": {"test_auc": 0.8}} for split_id in split_ids],
    }), encoding="utf-8")
    paired = build_paired_results(
        [f"F1,ctmp_gin,a3tgcn,{candidate_path},{reference_path}"],
        tmp_path / "eval.json",
    )
    assert paired["comparisons"][0]["same_graph_config"] is False


def test_build_paired_results_preserves_variable_split_sizes_for_nb_correction(tmp_path):
    split_artifact = {
        "splits": [
            {"split_id": "seed1_fold0", "eval_seed": 1, "fold": 0, "train_idx": [0, 1], "val_idx": [2], "test_idx": [3]},
            {"split_id": "seed1_fold1", "eval_seed": 1, "fold": 1, "train_idx": [0, 1, 2], "val_idx": [3], "test_idx": [4]},
        ]
    }
    split_path = tmp_path / "eval.json"
    split_path.write_text(json.dumps(split_artifact), encoding="utf-8")
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "reference.json"
    for path, auc in [(candidate_path, 0.9), (reference_path, 0.8)]:
        path.write_text(json.dumps({
            "graph_config_fingerprint": "graph123",
            "results": [
                {"split_id": "seed1_fold0", "result": {"test_auc": auc}},
                {"split_id": "seed1_fold1", "result": {"test_auc": auc}},
            ],
        }), encoding="utf-8")
    paired = build_paired_results(
        [f"F1,ctmp_gin,a3tgcn,{candidate_path},{reference_path}"],
        split_path,
    )
    comparison = paired["comparisons"][0]
    assert comparison["n_train_values"] == [3, 4]
    assert comparison["n_test_values"] == [1, 1]
    assert comparison["split_sizes_constant"] is False
    assert comparison["nb_test_train_ratio"] == pytest.approx(((1 / 3) + (1 / 4)) / 2)
    paired_path = tmp_path / "paired.json"
    paired_path.write_text(json.dumps(paired), encoding="utf-8")
    analyzed = analyze_paired_results(str(paired_path))
    assert analyzed["comparisons"][0]["raw"]["test_train_ratio"] == pytest.approx(comparison["nb_test_train_ratio"])


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


def test_joint_hpo_graph_space_uses_conditional_threshold_names():
    optuna = pytest.importorskip("optuna")
    trial = optuna.create_study(direction="maximize").ask()
    cfg = {"model": {"name": "ctmp_gin", "params": {}}, "train": {}, "edge": {}}
    out = suggest_protocol_params(trial, cfg)
    assert "n_neighbors" not in trial.params
    assert "threshold" not in trial.params
    assert trial.params["score_method"] in {"raw_mi", "nmi"}
    active = "threshold_raw_mi" if trial.params["score_method"] == "raw_mi" else "threshold_nmi"
    inactive = "threshold_nmi" if active == "threshold_raw_mi" else "threshold_raw_mi"
    assert active in trial.params
    assert inactive not in trial.params
    graph = normalize_graph_params(trial.params)
    assert out["edge"]["threshold"] == graph["threshold"]
    assert out["train"]["optimizer"] == "adam"


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
