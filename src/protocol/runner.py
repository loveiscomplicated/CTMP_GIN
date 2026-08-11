from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import optuna
except ImportError:  # prepare/preflight should remain usable without HPO dependencies.
    optuna = None  # type: ignore[assignment]

from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.models.factory import build_edge
from src.trainers.run_single_experiment import run_single_experiment

from .artifacts import create_eval_artifact, create_hpo_artifact, load_json, validate_artifact
from .constants import EVAL_SEEDS, HPO_SEED
from .graph_config import hub_concentration, load_graph_config, write_graph_config
from .hpo import apply_trial_params, suggest_protocol_params
from .preflight import run_preflight
from .analysis import analyze_paired_results
from .ablations import VARIANTS, apply_variant


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_codebook_path(cfg: dict[str, Any]) -> str | None:
    return cfg.get("codebook_path") or (cfg.get("data") or {}).get("codebook_path")


def _require_protocol_codebook(cfg: dict[str, Any], stage: str) -> None:
    if stage == "analyze":
        return
    codebook_path = _resolve_codebook_path(cfg)
    if not codebook_path:
        raise SystemExit("--codebook is required for protocol stages that build datasets")
    if not Path(codebook_path).exists():
        raise SystemExit(f"codebook does not exist: {codebook_path}")
    cfg["codebook_path"] = codebook_path


def _dataset(cfg: dict[str, Any], root: str):
    model = cfg["model"]["name"]
    admission_only = bool(cfg.get("admission_only", False))
    discharge_only = bool(cfg.get("discharge_only", False))
    los_as_node = bool(cfg.get("los_as_node", False))
    remove_los = not (los_as_node or (not admission_only and (discharge_only or model in {"gin", "a3tgcn_2_points", "gin_gru_2_points"})))
    dataset = TEDSTensorDataset(
        root=root,
        binary=cfg.get("train", {}).get("binary", True),
        ig_label=cfg.get("train", {}).get("ig_label", False),
        remove_los=remove_los,
        do_preprocess=cfg.get("train", {}).get("do_preprocess", True),
        admission_only=admission_only,
        discharge_only=discharge_only,
        los_as_node=los_as_node,
        codebook_path=cfg.get("codebook_path") or (cfg.get("data") or {}).get("codebook_path"),
    )
    labels = np.asarray([int(dataset[index][1]) for index in range(len(dataset))], dtype=np.int64)
    return dataset, labels


def prepare_artifacts(cfg: dict[str, Any], root: str, run_dir: str) -> dict[str, Any]:
    dataset, labels = _dataset(cfg, root)
    run_path = Path(run_dir)
    eval_artifact = create_eval_artifact(labels, run_path / "d_eval_split_artifact.json")
    d_hpo_idx = np.asarray(eval_artifact["d_hpo_idx"], dtype=np.int64)
    hpo_artifact = create_hpo_artifact(
        labels[d_hpo_idx],
        run_path / "d_hpo_split_artifact.json",
        base_indices=d_hpo_idx,
    )
    report = {
        "dataset_size": len(labels),
        "d_hpo_size": len(eval_artifact["d_hpo_idx"]),
        "d_eval_size": len(eval_artifact["d_eval_idx"]),
        "eval_split_count": len(eval_artifact["splits"]),
        "hpo_subset_size": len(hpo_artifact["hpo_subset_idx"]),
        "hpo_subset_fold_count": len(hpo_artifact["subset_folds"]),
        "hpo_full_fold_count": len(hpo_artifact["full_folds"]),
        "split_seed": eval_artifact["split_seed"],
        "hpo_seed": hpo_artifact["hpo_seed"],
        "codebook_report": getattr(dataset, "codebook_report", None),
    }
    _write(run_path / "split_report.json", report)
    return {"eval": eval_artifact, "hpo": hpo_artifact, "report": report}


def _require_artifacts(run_dir: str):
    run_path = Path(run_dir)
    eval_artifact = load_json(run_path / "d_eval_split_artifact.json")
    hpo_artifact = load_json(run_path / "d_hpo_split_artifact.json")
    return run_path, eval_artifact, hpo_artifact


def _fold_indices(fold_info: dict[str, Any], *, test: bool = False):
    return (
        np.asarray(fold_info["train_idx"], dtype=np.int64),
        np.asarray(fold_info["val_idx"], dtype=np.int64),
        np.asarray(fold_info.get("test_idx", []), dtype=np.int64) if test else np.asarray([], dtype=np.int64),
    )


def _score_config(cfg: dict[str, Any], root: str, fold_info: dict[str, Any], trial_number: int, *, trial: optuna.Trial | None = None, test: bool = False) -> float:
    train_idx, val_idx, test_idx = _fold_indices(fold_info, test=test)
    if not test:
        # run_single_experiment still requires a test loader; validation-only HPO uses
        # the fold validation indices for both loader construction and ignores test AUC.
        test_idx = val_idx
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("train", {})["drop_last"] = True
    cfg["train"]["eval_drop_last"] = False
    cfg["train"]["seed"] = HPO_SEED + int(trial_number)
    cfg["train"]["split_seed"] = HPO_SEED
    result = run_single_experiment(
        cfg,
        root=root,
        trial=trial,
        suppress_logger=trial is None,
        split_indices=(train_idx, val_idx, test_idx),
        model_seed=HPO_SEED + int(trial_number),
    )
    return float(result.get("best_valid_metric", result.get("valid_auc", result.get("roc_auc", float("nan")))))


def run_hpo(
    cfg: dict[str, Any],
    root: str,
    run_dir: str,
    graph_config_path: str,
    n_trials: int = 100,
    study_name: str | None = None,
    warm_start_params: dict[str, Any] | None = None,
):
    if optuna is None:
        raise RuntimeError("Optuna is required for the hpo stage; install requirements.txt")
    run_path, _, hpo_artifact = _require_artifacts(run_dir)
    graph_config = load_graph_config(graph_config_path)
    storage = f"sqlite:///{(run_path / 'protocol_optuna.db').resolve()}"
    study_name = study_name or f"{cfg['model']['name']}_protocol"
    sampler = optuna.samplers.TPESampler(seed=HPO_SEED, multivariate=True, group=True, n_startup_trials=20, constant_liar=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    if warm_start_params and not study.trials:
        study.enqueue_trial(warm_start_params)
    folds = hpo_artifact["subset_folds"]

    def objective(trial: optuna.Trial):
        trial_cfg = suggest_protocol_params(trial, cfg)
        trial_cfg = apply_trial_params(trial_cfg, trial.params, graph_config)
        scores = []
        for fold_info in folds:
            score = _score_config(trial_cfg, root, fold_info, trial.number, trial=trial)
            if not np.isfinite(score):
                raise optuna.TrialPruned()
            scores.append(score)
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    _write(run_path / "hpo_summary.json", {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "completed": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "best_trial": study.best_trial.number if study.best_trial else None,
        "best_params": study.best_params if study.best_trial else {},
        "graph_config_fingerprint": graph_config["graph_config_fingerprint"],
    })
    return study


def run_top5(cfg: dict[str, Any], root: str, run_dir: str, graph_config_path: str, study_name: str) -> dict[str, Any]:
    if optuna is None:
        raise RuntimeError("Optuna is required for the top5-reeval stage; install requirements.txt")
    run_path, _, hpo_artifact = _require_artifacts(run_dir)
    graph_config = load_graph_config(graph_config_path)
    storage = f"sqlite:///{(run_path / 'protocol_optuna.db').resolve()}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = sorted(
        (trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE),
        key=lambda trial: float(trial.value),
        reverse=True,
    )[:5]
    rankings = []
    for trial in completed:
        trial_cfg = apply_trial_params(cfg, trial.params, graph_config)
        scores = [_score_config(trial_cfg, root, fold, trial.number) for fold in hpo_artifact["full_folds"]]
        rankings.append({"trial_number": trial.number, "subset_value": trial.value, "full_scores": scores, "full_mean": float(np.mean(scores)), "params": trial.params})
    if not rankings:
        raise RuntimeError("top-5 reevaluation requires at least one completed HPO trial")
    winner = max(rankings, key=lambda item: item["full_mean"])
    selected = {"trial_number": winner["trial_number"], "params": winner["params"], "full_mean": winner["full_mean"], "graph_config_fingerprint": graph_config["graph_config_fingerprint"]}
    _write(run_path / "top5_reevaluation.json", {"rankings": rankings, "winner": selected})
    _write(run_path / "selected_config.json", selected)
    return selected


def run_graph_pilot(cfg: dict[str, Any], root: str, run_dir: str, n_trials: int = 20) -> dict[str, Any]:
    if optuna is None:
        raise RuntimeError("Optuna is required for the edge-pilot stage; install requirements.txt")
    run_path, _, hpo_artifact = _require_artifacts(run_dir)
    pilot_dataset, _ = _dataset(cfg, root)
    pilot_results = []
    for score_method, thresholds, study_name in [
        ("raw_mi", [0.0, 0.005, 0.01, 0.02], "edge_pilot_raw_mi"),
        ("nmi", [0.0, 0.02, 0.05, 0.10], "edge_pilot_nmi"),
    ]:
        storage = f"sqlite:///{(run_path / 'protocol_optuna.db').resolve()}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=HPO_SEED, multivariate=True, group=True, constant_liar=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            load_if_exists=True,
        )

        def objective(trial: optuna.Trial):
            trial_cfg = copy.deepcopy(cfg)
            trial_cfg.setdefault("train", {})["optimizer"] = "adamw"
            trial_cfg["train"]["drop_last"] = True
            trial_cfg["train"]["eval_drop_last"] = False
            trial_cfg.setdefault("edge", {}).update({
                "is_mi_based": True,
                "score_method": score_method,
                "threshold": trial.suggest_categorical("threshold", thresholds),
                "top_k": trial.suggest_categorical("top_k", [3, 6, 9, 12]),
                "pruning_ratio": trial.suggest_categorical("pruning_ratio", [0.0, 0.3, 0.5, 0.7]),
            })
            scores = []
            concentrations = []
            for fold in hpo_artifact["full_folds"]:
                scores.append(_score_config(trial_cfg, root, fold, trial.number, trial=trial))
                train_idx = np.asarray(fold["train_idx"], dtype=np.int64)
                if cfg["model"]["name"] in {"gin", "mlp"}:
                    num_nodes = len(pilot_dataset.col_info[0])
                else:
                    num_nodes = len(pilot_dataset.col_info[2])
                edge_index = build_edge(
                    model_name=cfg["model"]["name"],
                    root=root,
                    seed=HPO_SEED,
                    train_df=pilot_dataset.processed_df.iloc[train_idx],
                    num_nodes=num_nodes,
                    batch_size=trial_cfg["train"]["batch_size"],
                    **trial_cfg["edge"],
                )
                concentrations.append(hub_concentration(edge_index, pilot_dataset.col_info[1][:num_nodes]))
            trial.set_user_attr("hub_concentration", float(np.mean(concentrations)))
            return float(np.mean(scores))

        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
        best = study.best_trial
        pilot_results.append({
            "study_name": study_name,
            "score_method": score_method,
            "value": best.value,
            "params": best.params,
            "hub_concentration": best.user_attrs.get("hub_concentration", 0.0),
        })
    # AUC wins when the arms differ by more than the protocol tolerance. If tied,
    # prefer the graph with lower hub concentration to reduce cardinality bias.
    pilot_results.sort(key=lambda item: float(item["value"]), reverse=True)
    selected = pilot_results[0]
    if len(pilot_results) == 2 and abs(float(pilot_results[0]["value"]) - float(pilot_results[1]["value"])) <= 0.002:
        selected = min(pilot_results, key=lambda item: float(item["hub_concentration"]))
    pilot_artifact = {"pilot_results": pilot_results, "selected": selected}
    pilot_artifact["artifact_fingerprint"] = __import__("hashlib").blake2b(json.dumps(pilot_artifact, sort_keys=True).encode("utf-8"), digest_size=12).hexdigest()
    _write(run_path / "edge_pilot.json", pilot_artifact)
    return write_graph_config(
        str(run_path / "graph_config.json"),
        {
            "score_method": selected["score_method"],
            "threshold": selected["params"]["threshold"],
            "top_k": selected["params"]["top_k"],
            "pruning_ratio": selected["params"]["pruning_ratio"],
            "pilot_study": selected["study_name"],
        },
        pilot_artifact,
    )


def run_evaluation(cfg: dict[str, Any], root: str, run_dir: str, graph_config_path: str, selected_config_path: str):
    run_path, eval_artifact, _ = _require_artifacts(run_dir)
    graph_config = load_graph_config(graph_config_path)
    selected = load_json(selected_config_path)
    selected_cfg = apply_trial_params(cfg, selected["params"], graph_config)
    results = []
    for split in eval_artifact["splits"]:
        train_idx, val_idx, test_idx = _fold_indices(split, test=True)
        split_cfg = copy.deepcopy(selected_cfg)
        split_cfg["train"]["seed"] = int(split["eval_seed"])
        split_cfg["train"]["split_seed"] = int(split["eval_seed"])
        split_cfg["train"]["drop_last"] = True
        split_cfg["train"]["eval_drop_last"] = False
        output = run_single_experiment(
            split_cfg,
            root=root,
            split_indices=(train_idx, val_idx, test_idx),
            model_seed=int(split["eval_seed"]),
            suppress_logger=True,
        )
        results.append({"split_id": split["split_id"], "result": output})
        _write(run_path / "evaluation" / f"{split['split_id']}.json", results[-1])
    _write(run_path / "evaluation_summary.json", {
        "count": len(results),
        "graph_config_fingerprint": graph_config["graph_config_fingerprint"],
        "results": results,
    })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="CTMP-GIN reproducible protocol runner")
    parser.add_argument("--stage", required=True, choices=["preflight", "prepare", "edge-pilot", "hpo", "top5-reeval", "evaluate", "ablation-hpo", "ablation-evaluate", "analyze"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--graph-config", default=None)
    parser.add_argument("--selected-config", default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--codebook", default=None)
    parser.add_argument("--paired-results", default=None)
    parser.add_argument("--sesoi", type=float, default=None)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--warm-start", default=None)
    args = parser.parse_args()
    cfg = _load_cfg(args.config)
    if args.codebook:
        if not Path(args.codebook).exists():
            raise SystemExit(f"codebook does not exist: {args.codebook}")
        cfg["codebook_path"] = args.codebook
    _require_protocol_codebook(cfg, args.stage)
    if args.stage == "preflight":
        _, labels = _dataset(cfg, args.root)
        run_preflight(args.run_dir, labels, require_graph=Path(args.run_dir, "graph_config.json").exists())
    elif args.stage == "prepare":
        prepare_artifacts(cfg, args.root, args.run_dir)
    elif args.stage == "edge-pilot":
        run_graph_pilot(cfg, args.root, args.run_dir, args.n_trials if args.n_trials != 100 else 20)
    elif args.stage == "hpo":
        if not args.graph_config:
            raise SystemExit("--graph-config is required for hpo")
        run_hpo(cfg, args.root, args.run_dir, args.graph_config, args.n_trials, args.study_name)
    elif args.stage == "ablation-hpo":
        if not args.graph_config or not args.variant or args.variant == "full":
            raise SystemExit("--graph-config and a non-full --variant are required for ablation-hpo")
        if not VARIANTS[args.variant].get("hpo", False):
            raise SystemExit(f"{args.variant} uses inherited HPO; run ablation-evaluate instead")
        warm_start = load_json(args.warm_start)["params"] if args.warm_start else None
        variant_cfg = apply_variant(cfg, args.variant)
        run_hpo(variant_cfg, args.root, args.run_dir, args.graph_config, args.n_trials if args.n_trials != 100 else 40, args.study_name or f"{cfg['model']['name']}_{args.variant}", warm_start)
    elif args.stage == "top5-reeval":
        if not args.graph_config or not args.study_name:
            raise SystemExit("--graph-config and --study-name are required for top5-reeval")
        run_top5(cfg, args.root, args.run_dir, args.graph_config, args.study_name)
    elif args.stage == "evaluate":
        if not args.graph_config or not args.selected_config:
            raise SystemExit("--graph-config and --selected-config are required for evaluate")
        run_evaluation(cfg, args.root, args.run_dir, args.graph_config, args.selected_config)
    elif args.stage == "ablation-evaluate":
        if not args.graph_config or not args.selected_config or not args.variant or args.variant == "full":
            raise SystemExit("--graph-config, --selected-config, and a non-full --variant are required")
        run_evaluation(apply_variant(cfg, args.variant), args.root, args.run_dir, args.graph_config, args.selected_config)
    elif args.stage == "analyze":
        if not args.paired_results:
            raise SystemExit("--paired-results is required for analyze")
        result = analyze_paired_results(args.paired_results, sesoi=args.sesoi)
        _write(Path(args.run_dir) / "statistical_analysis.json", result)


if __name__ == "__main__":
    main()
