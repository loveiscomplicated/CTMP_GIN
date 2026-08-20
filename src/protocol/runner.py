from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import yaml

try:
    import optuna
except ImportError:  # prepare/preflight should remain usable without HPO dependencies.
    optuna = None  # type: ignore[assignment]

from src.data_processing.tackle_missing_value import tackle_missing_value_wrapper
from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.models.factory import build_edge
from src.trainers.run_single_experiment import run_single_experiment

from .artifacts import create_eval_artifact, create_hpo_artifact, load_json, validate_artifact
from .constants import EVAL_SEEDS, HPO_SEED
from .graph_config import hub_concentration, load_graph_config, write_graph_config
from .hpo import apply_trial_params, normalize_graph_params, selected_config_fingerprint, suggest_protocol_params
from .preflight import run_preflight
from .analysis import analyze_paired_results, build_paired_results
from .ablations import VARIANTS, apply_variant, validate_ablation_mutation
from .vocabulary import write_codebook_from_csv, write_codebook_from_frame


DATA_STAGES = {
    "preflight",
    "prepare",
    "edge-pilot",
    "hpo",
    "top5-plan",
    "top5-score",
    "top5-reeval",
    "evaluate",
    "ablation-hpo",
    "ablation-evaluate",
}

TOP_REEVAL_FOLD_COUNT = 3
TOP_REEVAL_MANIFEST = "top5_reevaluation_manifest.json"
TOP_REEVAL_SCORE_DIR = "top5_reevaluation_scores"

CTMP_VARIANTS = {
    "A1",
    "A2",
    "A3",
    "A4",
    "B1",
    "B3",
    "w/o_merged_stream",
    "w/o_gated_fusion",
    "w/o_mi_edge",
    "w/o_preprocessing",
}


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    tmp_path.replace(path)


def _config_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.blake2b(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"),
        digest_size=12,
    ).hexdigest()


def _load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _default_sqlite_storage(run_path: Path) -> str:
    return f"sqlite:///{(run_path / 'protocol_optuna.db').resolve()}"


def _storage_backend(storage: str) -> str:
    return urlsplit(storage).scheme.split("+", 1)[0].lower()


def _is_sqlite_storage(storage: str) -> bool:
    return _storage_backend(storage) == "sqlite"


def _redact_storage(storage: str) -> str:
    parsed = urlsplit(storage)
    if not parsed.password:
        return storage
    user = parsed.username or ""
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{user}:***@{host}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _resolve_optuna_storage(
    run_path: Path,
    storage: str | None = None,
    *,
    allow_sqlite_storage: bool = False,
) -> str:
    resolved = storage or os.environ.get("PROTOCOL_OPTUNA_STORAGE") or os.environ.get("OPTUNA_STORAGE")
    if resolved is None:
        if not allow_sqlite_storage:
            raise SystemExit(
                "Optuna stages require --storage or PROTOCOL_OPTUNA_STORAGE. "
                "Use PostgreSQL for parallel HPO. For local single-process smoke runs, "
                "pass --allow-sqlite-storage explicitly."
            )
        resolved = _default_sqlite_storage(run_path)
    if _is_sqlite_storage(resolved) and not allow_sqlite_storage:
        raise SystemExit(
            "SQLite Optuna storage is disabled by default for protocol HPO because it is "
            "not safe for multi-worker GPU runs. Use PostgreSQL, or pass "
            "--allow-sqlite-storage for an explicit local single-process run."
        )
    return resolved


def _optuna_storage_handle(
    storage: str,
    *,
    heartbeat_interval: int | None = None,
    heartbeat_grace_period: int | None = None,
):
    if (
        optuna is not None
        and heartbeat_interval is not None
        and int(heartbeat_interval) > 0
        and _storage_backend(storage) in {"postgres", "postgresql"}
    ):
        return optuna.storages.RDBStorage(
            url=storage,
            heartbeat_interval=int(heartbeat_interval),
            grace_period=heartbeat_grace_period,
        )
    return storage


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unnamed"


def _default_study_prefix(run_path: Path) -> str:
    resolved = str(run_path.resolve())
    digest = hashlib.blake2b(resolved.encode("utf-8"), digest_size=6).hexdigest()
    return f"{_safe_filename(run_path.name)}_{digest}"


def _resolve_study_prefix(run_path: Path, study_prefix: str | None = None) -> str:
    return _safe_filename(study_prefix) if study_prefix else _default_study_prefix(run_path)


def _namespaced_study_name(run_path: Path, study_name: str, study_prefix: str | None = None) -> str:
    return f"{_resolve_study_prefix(run_path, study_prefix)}__{_safe_filename(study_name)}"


def _worker_id() -> str:
    parts = [
        socket.gethostname(),
        f"pid{os.getpid()}",
        f"gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}",
    ]
    return _safe_filename("_".join(parts))


def _study_trial_count(study: optuna.Study) -> int:
    return len(study.get_trials(deepcopy=False))


def _trial_state_counts(study: optuna.Study) -> dict[str, int]:
    states = optuna.trial.TrialState
    trials = study.get_trials(deepcopy=False)
    counts = {
        "complete": sum(trial.state == states.COMPLETE for trial in trials),
        "running": sum(trial.state == states.RUNNING for trial in trials),
        "pruned": sum(trial.state == states.PRUNED for trial in trials),
        "failed": sum(trial.state == states.FAIL for trial in trials),
        "waiting": sum(trial.state == states.WAITING for trial in trials),
    }
    counts["attempted"] = counts["complete"] + counts["running"] + counts["pruned"] + counts["failed"]
    counts["total"] = len(trials)
    return counts


def _discord_trial_callback(
    *,
    enabled: bool,
    bot_name: str,
    max_total_trials: int | None,
):
    if not enabled:
        return None

    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        from src.utils.send_message import send_discord_message

        completed = sum(
            item.state == optuna.trial.TrialState.COMPLETE
            for item in study.get_trials(deepcopy=False)
        )
        target = str(max_total_trials) if max_total_trials is not None else "unknown"
        value = "pruned" if trial.value is None else f"{float(trial.value):.6f}"
        message = (
            f"[HPO_TRIAL_DONE] study={study.study_name} trial={trial.number} "
            f"state={trial.state.name} value={value} completed={completed}/{target} "
            f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
        )
        if not send_discord_message(message, bot_name=bot_name):
            print("Warning: Discord notification failed for HPO trial; continuing.", file=sys.stderr)

    return callback


def _budget_callback(target_completed_trials: int | None, max_total_trials: int | None):
    if target_completed_trials is None and max_total_trials is None:
        return None

    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        counts = _trial_state_counts(study)
        if target_completed_trials is not None and counts["complete"] >= int(target_completed_trials):
            study.stop()
        if max_total_trials is not None and counts["attempted"] >= int(max_total_trials):
            study.stop()

    return callback


def _optimize_study(
    study: optuna.Study,
    objective,
    *,
    n_trials: int,
    target_completed_trials: int | None = None,
    max_total_trials: int | None = None,
    notify_trials: bool = False,
    discord_bot_name: str = "protocol_runner",
) -> None:
    if n_trials <= 0:
        return
    callbacks = [
        callback
        for callback in (
            _budget_callback(target_completed_trials, max_total_trials),
            _discord_trial_callback(
                enabled=notify_trials,
                bot_name=discord_bot_name,
                max_total_trials=target_completed_trials or max_total_trials,
            ),
        )
        if callback is not None
    ]
    counts = _trial_state_counts(study)
    effective_trials = int(n_trials)
    if target_completed_trials is not None:
        remaining_completed_slots = int(target_completed_trials) - counts["complete"] - counts["running"]
        if remaining_completed_slots <= 0:
            return
        effective_trials = min(effective_trials, remaining_completed_slots)
    if max_total_trials is not None:
        remaining_attempts = int(max_total_trials) - counts["attempted"]
        if remaining_attempts <= 0:
            return
        effective_trials = min(effective_trials, remaining_attempts)
    study.optimize(
        objective,
        n_trials=effective_trials,
        callbacks=callbacks,
        gc_after_trial=True,
    )


def _resolve_codebook_path(cfg: dict[str, Any]) -> str | None:
    return cfg.get("codebook_path") or (cfg.get("data") or {}).get("codebook_path")


def _write_auto_protocol_codebook(root: str, cfg: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    raw_dir = Path(root) / "raw"
    raw_data_path = raw_dir / "TEDS_Discharge.csv"
    missing_corrected_path = raw_dir / "missing_corrected.csv"
    if cfg.get("train", {}).get("do_preprocess", True):
        frame = tackle_missing_value_wrapper(str(raw_data_path), str(missing_corrected_path))
        return write_codebook_from_frame(frame, output_path, source_csv=missing_corrected_path)
    return write_codebook_from_csv(raw_data_path, output_path)


def _auto_codebook_target(run_dir: str, variant: str = "full") -> Path:
    if variant == "full":
        return Path(run_dir) / "codebook.json"
    return Path(run_dir) / "codebooks" / f"{_safe_filename(variant)}.json"


def _ensure_protocol_codebook(
    cfg: dict[str, Any],
    stage: str,
    root: str | None,
    run_dir: str | None,
    *,
    variant: str = "full",
) -> None:
    if stage not in DATA_STAGES:
        return
    codebook_path = _resolve_codebook_path(cfg)
    if codebook_path and not Path(codebook_path).exists():
        raise SystemExit(f"codebook does not exist: {codebook_path}")
    if not codebook_path:
        if not root or not run_dir:
            raise SystemExit("--root and --run-dir are required to auto-generate protocol codebook")
        codebook_target = _auto_codebook_target(run_dir, variant)
        if codebook_target.exists():
            resolved = str(codebook_target.resolve())
        else:
            report = _write_auto_protocol_codebook(root, cfg, codebook_target)
            resolved = report["path"]
        cfg.setdefault("data", {})["codebook_path"] = resolved
        cfg["codebook_path"] = resolved
        return
    cfg["codebook_path"] = codebook_path


def _require_protocol_codebook(cfg: dict[str, Any], stage: str) -> None:
    if stage not in DATA_STAGES:
        return
    codebook_path = _resolve_codebook_path(cfg)
    if not codebook_path:
        raise SystemExit("--codebook is required for protocol stages that build datasets")
    if not Path(codebook_path).exists():
        raise SystemExit(f"codebook does not exist: {codebook_path}")
    cfg["codebook_path"] = codebook_path


def _required_model_for_variant(variant: str) -> str | None:
    if variant == "full":
        return None
    if variant in CTMP_VARIANTS:
        return "ctmp_gin"
    source = VARIANTS.get(variant, {}).get("source")
    return str(source) if source else None


def _validate_variant_config(cfg: dict[str, Any], variant: str) -> None:
    if variant not in VARIANTS:
        raise SystemExit(f"unknown protocol variant: {variant}")
    required_model = _required_model_for_variant(variant)
    actual_model = cfg.get("model", {}).get("name")
    if required_model and actual_model != required_model:
        raise SystemExit(
            f"{variant} must be run with a {required_model} config, got {actual_model!r}"
        )


def _variant_cfg(cfg: dict[str, Any], variant: str) -> dict[str, Any]:
    _validate_variant_config(cfg, variant)
    return cfg if variant == "full" else apply_variant(cfg, variant)


def _validate_selected_config_for_variant(
    selected: dict[str, Any],
    cfg: dict[str, Any],
    variant: str,
    selected_config_path: str,
) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown protocol variant: {variant}")
    selected_model = selected.get("model_name")
    actual_model = cfg.get("model", {}).get("name")
    if selected_model and actual_model and selected_model != actual_model:
        raise ValueError(
            f"{selected_config_path} was selected for model {selected_model!r}, "
            f"but current config uses {actual_model!r}"
        )

    selected_variant = str(selected.get("variant", "full"))
    if variant == "full":
        if selected_variant != "full":
            raise ValueError(
                f"full evaluate requires a full selected config, got variant {selected_variant!r}"
            )
        return

    if VARIANTS[variant].get("hpo", False) and selected_variant != variant:
        raise ValueError(
            f"{variant} requires its own top5-selected config from ablation-hpo/top5-reeval; "
            f"{selected_config_path} has variant {selected_variant!r}"
        )

    if selected_variant not in {variant, "full"}:
        raise ValueError(
            f"{selected_config_path} has variant {selected_variant!r}, "
            f"which cannot be used for evaluating {variant!r}"
        )


def _load_warm_start_params(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = load_json(path)
    selected = payload.get("selected", payload)
    graph = {
        **selected.get("params", {}),
        **selected.get("graph_params", {}),
    }
    if "score_method" not in graph and "score_method" in selected:
        graph["score_method"] = selected["score_method"]
    score_method = str(graph.get("score_method", "raw_mi"))
    params = {
        "score_method": score_method,
        "top_k": int(graph["top_k"]),
        "pruning_ratio": float(graph["pruning_ratio"]),
    }
    threshold_key = "threshold_raw_mi" if score_method == "raw_mi" else "threshold_nmi"
    if "threshold" in graph:
        params[threshold_key] = float(graph["threshold"])
    elif threshold_key in graph:
        params[threshold_key] = float(graph[threshold_key])
    return params


def _apply_selected_config(cfg: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    params = selected.get("params", {})
    graph_params = selected.get("graph_params")
    return apply_trial_params(cfg, params, graph_params)


def _evaluation_metadata(
    *,
    cfg: dict[str, Any],
    selected: dict[str, Any],
    selected_cfg: dict[str, Any],
    variant: str,
    ablation_mutation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph_params = selected.get("graph_params")
    if graph_params is None and selected.get("params"):
        graph_params = normalize_graph_params(selected["params"])
    graph_fingerprint = _config_fingerprint(graph_params or {})
    selected_fingerprint = selected.get("config_fingerprint") or _config_fingerprint({
        "model_name": cfg.get("model", {}).get("name"),
        "variant": selected.get("variant", "full"),
        "params": selected.get("params", {}),
        "graph_params": graph_params,
    })
    effective_fingerprint = _config_fingerprint(selected_cfg)
    metadata = {
        "variant": variant,
        "model_name": cfg["model"]["name"],
        "selected_config_fingerprint": selected_fingerprint,
        "effective_config_fingerprint": effective_fingerprint,
        "config_fingerprint": effective_fingerprint,
        "graph_config_fingerprint": graph_fingerprint,
        "graph_params": graph_params or {},
    }
    if variant != "full":
        metadata["parent_config_fingerprint"] = selected_fingerprint
        metadata["ablation_mutation"] = ablation_mutation
    return metadata


def _parse_eval_seeds(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip() == "":
        return None
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("--eval-seeds did not contain any valid seed")
    return seeds


def _parse_eval_split_ids(values: list[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    split_ids = []
    for value in values:
        split_ids.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(split_ids) if split_ids else None


def _select_eval_splits(
    eval_artifact: dict[str, Any],
    eval_seeds: tuple[int, ...] | None = None,
    eval_split_ids: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    splits = list(eval_artifact.get("splits", []))
    if eval_seeds is None and eval_split_ids is None:
        return splits
    selected = splits
    if eval_seeds is not None:
        available = {int(split["eval_seed"]) for split in splits}
        missing = sorted(set(eval_seeds) - available)
        if missing:
            raise ValueError(f"requested eval seeds not present in split artifact: {missing}")
        selected = [split for split in selected if int(split["eval_seed"]) in set(eval_seeds)]
    if eval_split_ids is not None:
        available_ids = {str(split["split_id"]) for split in splits}
        missing_ids = sorted(set(eval_split_ids) - available_ids)
        if missing_ids:
            raise ValueError(f"requested eval split_ids not present in split artifact: {missing_ids}")
        selected = [split for split in selected if str(split["split_id"]) in set(eval_split_ids)]
    if not selected:
        raise ValueError("no evaluation splits selected")
    return selected


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
    n_trials: int = 100,
    study_name: str | None = None,
    warm_start_params: dict[str, Any] | None = None,
    storage: str | None = None,
    allow_sqlite_storage: bool = False,
    study_prefix: str | None = None,
    variant: str = "full",
    target_completed_trials: int | None = None,
    max_total_trials: int | None = None,
    notify_trials: bool = False,
    discord_bot_name: str = "protocol_runner",
    heartbeat_interval: int | None = None,
    heartbeat_grace_period: int | None = None,
):
    if optuna is None:
        raise RuntimeError("Optuna is required for the hpo stage; install requirements.txt")
    run_path, _, hpo_artifact = _require_artifacts(run_dir)
    storage_url = _resolve_optuna_storage(run_path, storage, allow_sqlite_storage=allow_sqlite_storage)
    storage_handle = _optuna_storage_handle(
        storage_url,
        heartbeat_interval=heartbeat_interval,
        heartbeat_grace_period=heartbeat_grace_period,
    )
    requested_study_name = study_name or f"{cfg['model']['name']}_protocol"
    study_name = _namespaced_study_name(run_path, requested_study_name, study_prefix)
    sampler = optuna.samplers.TPESampler(seed=HPO_SEED, multivariate=True, group=True, n_startup_trials=20, constant_liar=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_handle,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    if warm_start_params and not study.trials:
        study.enqueue_trial(warm_start_params)
    folds = hpo_artifact["subset_folds"][:1]

    def objective(trial: optuna.Trial):
        trial_cfg = suggest_protocol_params(trial, cfg)
        trial_cfg = apply_trial_params(trial_cfg, trial.params)
        scores = []
        for fold_info in folds:
            score = _score_config(trial_cfg, root, fold_info, trial.number, trial=trial)
            if not np.isfinite(score):
                raise optuna.TrialPruned()
            scores.append(score)
        return float(np.mean(scores))

    _optimize_study(
        study,
        objective,
        n_trials=n_trials,
        target_completed_trials=target_completed_trials,
        max_total_trials=max_total_trials,
        notify_trials=notify_trials,
        discord_bot_name=discord_bot_name,
    )
    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    best_trial = study.best_trial if completed_trials else None
    counts = _trial_state_counts(study)
    summary = {
        "study_name": study.study_name,
        "requested_study_name": requested_study_name,
        "study_prefix": _resolve_study_prefix(run_path, study_prefix),
        "variant": variant,
        "worker_id": _worker_id(),
        "storage_backend": _storage_backend(storage_url),
        "storage": _redact_storage(storage_url),
        "n_trials": counts["total"],
        "target_completed_trials": target_completed_trials,
        "max_total_trials": max_total_trials,
        "completed": counts["complete"],
        "running": counts["running"],
        "pruned": counts["pruned"],
        "failed": counts["failed"],
        "waiting": counts["waiting"],
        "attempted": counts["attempted"],
        "best_trial": best_trial.number if best_trial else None,
        "best_params": study.best_params if best_trial else {},
        "sampler": {"name": "TPESampler", "multivariate": True, "group": True, "constant_liar": True},
        "hpo_fold_ids": [fold["fold"] for fold in folds],
    }
    _write(run_path / "hpo_summary.json", summary)
    _write(run_path / "hpo_summaries" / f"{_safe_filename(study.study_name)}_{summary['worker_id']}.json", summary)
    return study


def top5_manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / TOP_REEVAL_MANIFEST


def top5_score_path(run_dir: str | Path, trial_number: int, fold_index: int) -> Path:
    return (
        Path(run_dir)
        / TOP_REEVAL_SCORE_DIR
        / f"trial_{int(trial_number):04d}_fold_{int(fold_index):02d}.json"
    )


def _top5_manifest_fingerprint(manifest: dict[str, Any]) -> str:
    payload = copy.deepcopy(manifest)
    payload.pop("manifest_fingerprint", None)
    return _config_fingerprint(payload)


def _load_top5_manifest(run_path: Path, cfg: dict[str, Any] | None = None, variant: str | None = None) -> dict[str, Any]:
    manifest = load_json(top5_manifest_path(run_path))
    if manifest.get("artifact_type") != "top5_reevaluation_manifest":
        raise ValueError(f"invalid top5 manifest: {top5_manifest_path(run_path)}")
    expected_fingerprint = _top5_manifest_fingerprint(manifest)
    actual_fingerprint = manifest.get("manifest_fingerprint")
    if actual_fingerprint and actual_fingerprint != expected_fingerprint:
        raise ValueError(f"top5 manifest fingerprint mismatch: {top5_manifest_path(run_path)}")
    if cfg is not None:
        expected_model = cfg.get("model", {}).get("name")
        if manifest.get("model_name") != expected_model:
            raise ValueError(
                f"top5 manifest was created for model {manifest.get('model_name')!r}, "
                f"but current config uses {expected_model!r}"
            )
    if variant is not None and manifest.get("variant", "full") != variant:
        raise ValueError(
            f"top5 manifest was created for variant {manifest.get('variant', 'full')!r}, "
            f"but current variant is {variant!r}"
        )
    if not manifest.get("candidate_trials"):
        raise ValueError("top5 manifest has no candidate trials")
    return manifest


def _top5_candidate(manifest: dict[str, Any], trial_number: int) -> dict[str, Any]:
    for candidate in manifest.get("candidate_trials", []):
        if int(candidate["trial_number"]) == int(trial_number):
            return candidate
    raise ValueError(f"trial {trial_number} is not in the top5 reevaluation manifest")


def _load_top5_score(
    run_dir: str | Path,
    manifest: dict[str, Any],
    trial_number: int,
    fold_index: int,
) -> dict[str, Any] | None:
    path = top5_score_path(run_dir, trial_number, fold_index)
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        payload = load_json(path)
    except Exception:
        return None
    expected_fingerprint = manifest.get("manifest_fingerprint") or _top5_manifest_fingerprint(manifest)
    if payload.get("artifact_type") != "top5_reevaluation_score":
        return None
    if payload.get("manifest_fingerprint") != expected_fingerprint:
        return None
    if int(payload.get("trial_number", -1)) != int(trial_number):
        return None
    if int(payload.get("fold_index", -1)) != int(fold_index):
        return None
    try:
        if not np.isfinite(float(payload["score"])):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return payload


def top5_score_complete(
    run_dir: str | Path,
    manifest: dict[str, Any],
    trial_number: int,
    fold_index: int,
) -> bool:
    return _load_top5_score(run_dir, manifest, trial_number, fold_index) is not None


def top5_pending_scores(run_dir: str | Path) -> list[dict[str, int]]:
    run_path = Path(run_dir)
    manifest = _load_top5_manifest(run_path)
    pending = []
    fold_count = int(manifest["reeval_fold_count"])
    for candidate in manifest["candidate_trials"]:
        trial_number = int(candidate["trial_number"])
        for fold_index in range(fold_count):
            if not top5_score_complete(run_path, manifest, trial_number, fold_index):
                pending.append({"trial_number": trial_number, "fold_index": fold_index})
    return pending


def prepare_top5_reevaluation(
    cfg: dict[str, Any],
    run_dir: str,
    study_name: str | None = None,
    storage: str | None = None,
    allow_sqlite_storage: bool = False,
    study_prefix: str | None = None,
    variant: str = "full",
    top_n: int = 3,
) -> dict[str, Any]:
    if optuna is None:
        raise RuntimeError("Optuna is required for the top5-reeval stage; install requirements.txt")
    run_path, _, hpo_artifact = _require_artifacts(run_dir)
    manifest_path = top5_manifest_path(run_path)
    if manifest_path.exists():
        manifest = _load_top5_manifest(run_path, cfg, variant)
        if int(manifest["top_n"]) != int(top_n):
            raise ValueError(
                f"existing top5 manifest uses top_n={manifest['top_n']}, requested top_n={top_n}"
            )
        return manifest

    storage_url = _resolve_optuna_storage(run_path, storage, allow_sqlite_storage=allow_sqlite_storage)
    requested_study_name = study_name or (
        f"{cfg['model']['name']}_{variant}" if variant != "full" else f"{cfg['model']['name']}_protocol"
    )
    study_name = _namespaced_study_name(run_path, requested_study_name, study_prefix)
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    completed = sorted(
        (trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE),
        key=lambda trial: float(trial.value),
        reverse=True,
    )[:top_n]
    if not completed:
        raise RuntimeError("top-3 reevaluation requires at least one completed HPO trial")

    folds = hpo_artifact["full_folds"][:TOP_REEVAL_FOLD_COUNT]
    manifest = {
        "artifact_type": "top5_reevaluation_manifest",
        "model_name": cfg["model"]["name"],
        "variant": variant,
        "study_name": study.study_name,
        "requested_study_name": requested_study_name,
        "study_prefix": _resolve_study_prefix(run_path, study_prefix),
        "storage_backend": _storage_backend(storage_url),
        "storage": _redact_storage(storage_url),
        "top_n": int(top_n),
        "reeval_fold_count": len(folds),
        "fold_indices": list(range(len(folds))),
        "fold_ids": [int(fold["fold"]) for fold in folds],
        "hpo_artifact_fingerprint": hpo_artifact.get("artifact_fingerprint"),
        "candidate_trials": [
            {
                "trial_number": int(trial.number),
                "subset_value": float(trial.value),
                "params": dict(trial.params),
                "graph_params": normalize_graph_params(trial.params),
            }
            for trial in completed
        ],
        "selection_rule": "max reeval_mean, then lower reeval_std, then higher hpo_subset_value, then lower trial_number",
    }
    manifest["manifest_fingerprint"] = _top5_manifest_fingerprint(manifest)
    _write(manifest_path, manifest)
    return manifest


def run_top5_score(
    cfg: dict[str, Any],
    root: str,
    run_dir: str,
    trial_number: int,
    fold_index: int,
    *,
    variant: str = "full",
) -> dict[str, Any]:
    run_path, _, hpo_artifact = _require_artifacts(run_dir)
    manifest = _load_top5_manifest(run_path, cfg, variant)
    existing = _load_top5_score(run_path, manifest, trial_number, fold_index)
    if existing is not None:
        return existing

    fold_count = int(manifest["reeval_fold_count"])
    if fold_index < 0 or fold_index >= fold_count:
        raise ValueError(f"fold_index must be in [0, {fold_count}), got {fold_index}")
    candidate = _top5_candidate(manifest, trial_number)
    folds = hpo_artifact["full_folds"][:fold_count]
    if len(folds) < fold_count:
        raise ValueError(f"expected {fold_count} top5 reevaluation folds, found {len(folds)}")
    fold_info = folds[fold_index]
    trial_cfg = apply_trial_params(cfg, candidate["params"], candidate["graph_params"])
    score = _score_config(trial_cfg, root, fold_info, int(trial_number))
    if not np.isfinite(float(score)):
        raise RuntimeError(f"top5 reevaluation produced a non-finite score for trial={trial_number} fold={fold_index}")
    payload = {
        "artifact_type": "top5_reevaluation_score",
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "model_name": manifest["model_name"],
        "variant": manifest["variant"],
        "trial_number": int(trial_number),
        "fold_index": int(fold_index),
        "fold_id": int(fold_info["fold"]),
        "score": float(score),
        "params": candidate["params"],
        "graph_params": candidate["graph_params"],
    }
    _write(top5_score_path(run_path, trial_number, fold_index), payload)
    return payload


def finalize_top5_reevaluation(
    cfg: dict[str, Any],
    run_dir: str,
    *,
    variant: str = "full",
) -> dict[str, Any]:
    run_path, _, _ = _require_artifacts(run_dir)
    manifest = _load_top5_manifest(run_path, cfg, variant)
    rankings = []
    missing = []
    fold_count = int(manifest["reeval_fold_count"])
    for candidate in manifest["candidate_trials"]:
        trial_number = int(candidate["trial_number"])
        scores = []
        for fold_index in range(fold_count):
            score_payload = _load_top5_score(run_path, manifest, trial_number, fold_index)
            if score_payload is None:
                missing.append(f"trial={trial_number} fold_index={fold_index}")
                continue
            scores.append(float(score_payload["score"]))
        if len(scores) != fold_count:
            continue
        rankings.append({
            "trial_number": trial_number,
            "subset_value": float(candidate["subset_value"]),
            "reeval_scores": scores,
            "reeval_mean": float(np.mean(scores)),
            "reeval_std": float(np.std(scores)),
            "full_scores": scores,
            "full_mean": float(np.mean(scores)),
            "params": candidate["params"],
            "graph_params": candidate["graph_params"],
        })
    if missing:
        raise FileNotFoundError(f"missing top5 reevaluation score outputs: {missing}")
    if not rankings:
        raise RuntimeError("top-3 reevaluation requires at least one completed HPO trial")

    winner = max(
        rankings,
        key=lambda item: (
            item["reeval_mean"],
            -item["reeval_std"],
            item["subset_value"],
            -item["trial_number"],
        ),
    )
    config_fingerprint = selected_config_fingerprint(
        model_name=cfg["model"]["name"],
        variant=variant,
        params=winner["params"],
        graph_params=winner["graph_params"],
    )
    selected = {
        "trial_number": winner["trial_number"],
        "params": winner["params"],
        "graph_params": winner["graph_params"],
        "config_fingerprint": config_fingerprint,
        "hpo_subset_value": winner["subset_value"],
        "reeval_scores": winner["reeval_scores"],
        "reeval_mean": winner["reeval_mean"],
        "reeval_std": winner["reeval_std"],
        "full_mean": winner["reeval_mean"],
        "study_name": manifest["study_name"],
        "requested_study_name": manifest["requested_study_name"],
        "study_prefix": manifest["study_prefix"],
        "variant": variant,
        "model_name": cfg["model"]["name"],
        "selection_rule": manifest["selection_rule"],
    }
    payload = {
        "rankings": rankings,
        "winner": selected,
        "top_n": manifest["top_n"],
        "reeval_fold_count": fold_count,
        "manifest_fingerprint": manifest["manifest_fingerprint"],
    }
    _write(run_path / "top3_reevaluation.json", payload)
    _write(run_path / "top5_reevaluation.json", payload)
    _write(run_path / "selected_config.json", selected)
    if variant != "full":
        _write(run_path / f"selected_config_{_safe_filename(variant)}.json", selected)
    return selected


def run_top5(
    cfg: dict[str, Any],
    root: str,
    run_dir: str,
    study_name: str | None = None,
    storage: str | None = None,
    allow_sqlite_storage: bool = False,
    study_prefix: str | None = None,
    variant: str = "full",
    top_n: int = 3,
) -> dict[str, Any]:
    prepare_top5_reevaluation(
        cfg,
        run_dir,
        study_name,
        storage=storage,
        allow_sqlite_storage=allow_sqlite_storage,
        study_prefix=study_prefix,
        variant=variant,
        top_n=top_n,
    )
    for item in top5_pending_scores(run_dir):
        run_top5_score(
            cfg,
            root,
            run_dir,
            item["trial_number"],
            item["fold_index"],
            variant=variant,
        )
    return finalize_top5_reevaluation(cfg, run_dir, variant=variant)


def run_graph_pilot(
    cfg: dict[str, Any],
    root: str,
    run_dir: str,
    n_trials: int = 20,
    storage: str | None = None,
    allow_sqlite_storage: bool = False,
    study_prefix: str | None = None,
    max_total_trials: int | None = None,
    notify_trials: bool = False,
    discord_bot_name: str = "protocol_runner",
) -> dict[str, Any]:
    if optuna is None:
        raise RuntimeError("Optuna is required for the edge-pilot stage; install requirements.txt")
    run_path, _, hpo_artifact = _require_artifacts(run_dir)
    pilot_dataset, _ = _dataset(cfg, root)
    resolved_storage = _resolve_optuna_storage(run_path, storage, allow_sqlite_storage=allow_sqlite_storage)
    resolved_study_prefix = _resolve_study_prefix(run_path, study_prefix)
    pilot_results = []
    for score_method, thresholds, requested_study_name in [
        ("raw_mi", [0.0, 0.005, 0.01, 0.02], "edge_pilot_raw_mi"),
        ("nmi", [0.0, 0.02, 0.05, 0.10], "edge_pilot_nmi"),
    ]:
        study_name = _namespaced_study_name(run_path, requested_study_name, resolved_study_prefix)
        study = optuna.create_study(
            study_name=study_name,
            storage=resolved_storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=HPO_SEED, multivariate=True, group=True, constant_liar=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            load_if_exists=True,
        )

        def objective(trial: optuna.Trial):
            trial_cfg = copy.deepcopy(cfg)
            trial_cfg.setdefault("train", {})["optimizer"] = "adam"
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

        _optimize_study(
            study,
            objective,
            n_trials=n_trials,
            max_total_trials=max_total_trials,
            notify_trials=notify_trials,
            discord_bot_name=discord_bot_name,
        )
        completed_trials = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            raise RuntimeError(f"edge-pilot study {study.study_name} has no completed trials")
        best = study.best_trial
        pilot_results.append({
            "study_name": study_name,
            "requested_study_name": requested_study_name,
            "study_prefix": resolved_study_prefix,
            "score_method": score_method,
            "storage_backend": _storage_backend(resolved_storage),
            "max_total_trials": max_total_trials,
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
    pilot_artifact["model_name"] = cfg["model"]["name"]
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
        model_name=cfg["model"]["name"],
    )


def run_evaluation(
    cfg: dict[str, Any],
    root: str,
    run_dir: str,
    selected_config_path: str,
    *,
    eval_seeds: tuple[int, ...] | None = None,
    eval_split_ids: tuple[str, ...] | None = None,
    variant: str = "full",
    write_summary: bool = True,
):
    run_path, eval_artifact, _ = _require_artifacts(run_dir)
    selected = load_json(selected_config_path)
    _validate_selected_config_for_variant(selected, cfg, variant, selected_config_path)
    parent_cfg = _apply_selected_config(cfg, selected)
    ablation_mutation = None
    if variant == "full":
        selected_cfg = parent_cfg
    else:
        selected_cfg = apply_variant(parent_cfg, variant)
        ablation_mutation = validate_ablation_mutation(parent_cfg, selected_cfg, variant)
    metadata = _evaluation_metadata(
        cfg=cfg,
        selected=selected,
        selected_cfg=selected_cfg,
        variant=variant,
        ablation_mutation=ablation_mutation,
    )
    results = []
    splits = _select_eval_splits(eval_artifact, eval_seeds, eval_split_ids)

    for split in splits:
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
        results.append({"split_id": split["split_id"], "result": output, **metadata})
        _write(run_path / "evaluation" / f"{split['split_id']}.json", results[-1])
    if write_summary:
        _write(run_path / "evaluation_summary.json", {
            "count": len(results),
            "eval_seeds": sorted({int(split["eval_seed"]) for split in splits}),
            "results": results,
            **metadata,
        })
    return results


def finalize_evaluation(
    cfg: dict[str, Any],
    run_dir: str,
    *,
    eval_seeds: tuple[int, ...] | None = None,
    eval_split_ids: tuple[str, ...] | None = None,
    variant: str = "full",
) -> dict[str, Any]:
    run_path, eval_artifact, _ = _require_artifacts(run_dir)
    splits = _select_eval_splits(eval_artifact, eval_seeds, eval_split_ids)
    results = []
    missing = []
    for split in splits:
        split_id = str(split["split_id"])
        path = run_path / "evaluation" / f"{split_id}.json"
        if not path.exists():
            missing.append(split_id)
            continue
        results.append(load_json(path))
    if missing:
        raise FileNotFoundError(f"missing evaluation split outputs: {missing}")
    metadata_keys = {
        "model_name",
        "variant",
        "selected_config_fingerprint",
        "parent_config_fingerprint",
        "effective_config_fingerprint",
        "config_fingerprint",
        "graph_config_fingerprint",
        "graph_params",
        "ablation_mutation",
    }
    metadata = {
        key: results[0][key]
        for key in metadata_keys
        if results and key in results[0]
    }
    summary = {
        "count": len(results),
        "variant": variant,
        "model_name": cfg["model"]["name"],
        "eval_seeds": sorted({int(split["eval_seed"]) for split in splits}),
        "results": results,
        **metadata,
    }
    _write(run_path / "evaluation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CTMP-GIN reproducible protocol runner")
    parser.add_argument("--stage", required=True, choices=["preflight", "prepare", "edge-pilot", "hpo", "top5-plan", "top5-score", "top5-reeval", "evaluate", "finalize-evaluate", "ablation-hpo", "ablation-evaluate", "pair-results", "analyze"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--graph-config", default=None)
    parser.add_argument("--selected-config", default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--codebook", default=None)
    parser.add_argument("--paired-results", default=None)
    parser.add_argument("--comparison", action="append", default=[], help="For pair-results: family,candidate,reference,candidate_summary,reference_summary")
    parser.add_argument("--metric", default="test_auc")
    parser.add_argument("--split-artifact", default=None)
    parser.add_argument("--sesoi", type=float, default=None)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--warm-start", default=None)
    parser.add_argument("--eval-seeds", default=None, help="Comma-separated eval seeds to run, e.g. '1,2'. Defaults to all artifact seeds.")
    parser.add_argument("--eval-split-id", action="append", default=[], help="Evaluation split_id(s) to run/finalize. Can be repeated or comma-separated.")
    parser.add_argument("--no-summary", action="store_true", help="For evaluate stages, write split files only and skip evaluation_summary.json.")
    parser.add_argument("--top-n", type=int, default=3, help="Number of completed HPO trials to re-evaluate.")
    parser.add_argument("--reeval-trial-number", type=int, default=None, help="Top re-evaluation trial number for top5-score.")
    parser.add_argument("--reeval-fold-index", type=int, default=None, help="Top re-evaluation fold index for top5-score.")
    parser.add_argument("--finalize-only", action="store_true", help="For top5-reeval, only finalize existing score artifacts.")
    parser.add_argument("--study-prefix", default=None, help="Namespace Optuna study names. Defaults to a stable hash of --run-dir.")
    parser.add_argument("--target-completed-trials", type=int, default=40, help="Target COMPLETE trials for joint HPO.")
    parser.add_argument("--max-total-trials", type=int, default=80, help="Attempted-trial safety cap for multi-worker HPO.")
    parser.add_argument("--notify-trials", action="store_true", help="Send a Discord message after each HPO/edge-pilot trial.")
    parser.add_argument("--discord-bot-name", default="protocol_runner")
    parser.add_argument("--heartbeat-interval", type=int, default=60, help="PostgreSQL Optuna heartbeat interval in seconds; <=0 disables it.")
    parser.add_argument("--heartbeat-grace-period", type=int, default=180, help="PostgreSQL Optuna heartbeat grace period in seconds.")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL. Prefer PostgreSQL for parallel HPO. Can also be set via PROTOCOL_OPTUNA_STORAGE.",
    )
    parser.add_argument(
        "--allow-sqlite-storage",
        action="store_true",
        help="Explicitly allow local SQLite Optuna storage for single-process smoke runs.",
    )
    args = parser.parse_args()
    if args.stage in DATA_STAGES or args.stage == "finalize-evaluate":
        if not args.config:
            raise SystemExit("--config is required for data/model protocol stages")
        if args.stage in DATA_STAGES and not args.root:
            raise SystemExit("--root is required for data/model protocol stages")
        cfg = _load_cfg(args.config)
        if args.codebook:
            if not Path(args.codebook).exists():
                raise SystemExit(f"codebook does not exist: {args.codebook}")
            cfg["codebook_path"] = args.codebook
    else:
        cfg = {}
    if args.stage == "preflight":
        preflight_cfg = _variant_cfg(cfg, args.variant) if args.variant != "full" else cfg
        _ensure_protocol_codebook(preflight_cfg, args.stage, args.root, args.run_dir, variant=args.variant)
        _, labels = _dataset(preflight_cfg, args.root)
        run_preflight(args.run_dir, labels, require_graph=Path(args.run_dir, "graph_config.json").exists())
    elif args.stage == "prepare":
        prepare_cfg = _variant_cfg(cfg, args.variant) if args.variant != "full" else cfg
        _ensure_protocol_codebook(prepare_cfg, args.stage, args.root, args.run_dir, variant=args.variant)
        prepare_artifacts(prepare_cfg, args.root, args.run_dir)
    elif args.stage == "edge-pilot":
        _ensure_protocol_codebook(cfg, args.stage, args.root, args.run_dir)
        run_graph_pilot(
            cfg,
            args.root,
            args.run_dir,
            args.n_trials if args.n_trials != 100 else 20,
            storage=args.storage,
            allow_sqlite_storage=args.allow_sqlite_storage,
            study_prefix=args.study_prefix,
            max_total_trials=args.max_total_trials,
            notify_trials=args.notify_trials,
            discord_bot_name=args.discord_bot_name,
        )
    elif args.stage == "hpo":
        _ensure_protocol_codebook(cfg, args.stage, args.root, args.run_dir)
        run_hpo(
            cfg,
            args.root,
            args.run_dir,
            args.n_trials,
            args.study_name,
            warm_start_params=_load_warm_start_params(args.warm_start),
            storage=args.storage,
            allow_sqlite_storage=args.allow_sqlite_storage,
            study_prefix=args.study_prefix,
            target_completed_trials=args.target_completed_trials,
            max_total_trials=args.max_total_trials,
            notify_trials=args.notify_trials,
            discord_bot_name=args.discord_bot_name,
            heartbeat_interval=args.heartbeat_interval if args.heartbeat_interval > 0 else None,
            heartbeat_grace_period=args.heartbeat_grace_period,
        )
    elif args.stage == "ablation-hpo":
        if not args.variant or args.variant == "full":
            raise SystemExit("a non-full --variant is required for ablation-hpo")
        if not VARIANTS[args.variant].get("hpo", False):
            raise SystemExit(f"{args.variant} uses inherited HPO; run ablation-evaluate instead")
        if args.warm_start:
            raise SystemExit("ablation-hpo must run an independent Optuna search; --warm-start is disabled")
        warm_start = None
        variant_cfg = _variant_cfg(cfg, args.variant)
        _ensure_protocol_codebook(variant_cfg, args.stage, args.root, args.run_dir, variant=args.variant)
        run_hpo(
            variant_cfg,
            args.root,
            args.run_dir,
            args.n_trials if args.n_trials != 100 else 40,
            args.study_name or f"{cfg['model']['name']}_{args.variant}",
            warm_start,
            storage=args.storage,
            allow_sqlite_storage=args.allow_sqlite_storage,
            study_prefix=args.study_prefix,
            variant=args.variant,
            target_completed_trials=args.target_completed_trials,
            max_total_trials=args.max_total_trials,
            notify_trials=args.notify_trials,
            discord_bot_name=args.discord_bot_name,
            heartbeat_interval=args.heartbeat_interval if args.heartbeat_interval > 0 else None,
            heartbeat_grace_period=args.heartbeat_grace_period,
        )
    elif args.stage == "top5-plan":
        variant_cfg = _variant_cfg(cfg, args.variant)
        _ensure_protocol_codebook(variant_cfg, args.stage, args.root, args.run_dir, variant=args.variant)
        prepare_top5_reevaluation(
            variant_cfg,
            args.run_dir,
            args.study_name,
            storage=args.storage,
            allow_sqlite_storage=args.allow_sqlite_storage,
            study_prefix=args.study_prefix,
            variant=args.variant,
            top_n=args.top_n,
        )
    elif args.stage == "top5-score":
        if args.reeval_trial_number is None or args.reeval_fold_index is None:
            raise SystemExit("--reeval-trial-number and --reeval-fold-index are required for top5-score")
        variant_cfg = _variant_cfg(cfg, args.variant)
        _ensure_protocol_codebook(variant_cfg, args.stage, args.root, args.run_dir, variant=args.variant)
        run_top5_score(
            variant_cfg,
            args.root,
            args.run_dir,
            args.reeval_trial_number,
            args.reeval_fold_index,
            variant=args.variant,
        )
    elif args.stage == "top5-reeval":
        variant_cfg = _variant_cfg(cfg, args.variant)
        _ensure_protocol_codebook(variant_cfg, args.stage, args.root, args.run_dir, variant=args.variant)
        if args.finalize_only:
            finalize_top5_reevaluation(variant_cfg, args.run_dir, variant=args.variant)
        else:
            run_top5(
                variant_cfg,
                args.root,
                args.run_dir,
                args.study_name,
                storage=args.storage,
                allow_sqlite_storage=args.allow_sqlite_storage,
                study_prefix=args.study_prefix,
                variant=args.variant,
                top_n=args.top_n,
            )
    elif args.stage == "evaluate":
        if not args.selected_config:
            raise SystemExit("--selected-config is required for evaluate")
        _ensure_protocol_codebook(cfg, args.stage, args.root, args.run_dir)
        run_evaluation(
            cfg,
            args.root,
            args.run_dir,
            args.selected_config,
            eval_seeds=_parse_eval_seeds(args.eval_seeds),
            eval_split_ids=_parse_eval_split_ids(args.eval_split_id),
            variant="full",
            write_summary=not args.no_summary,
        )
    elif args.stage == "finalize-evaluate":
        variant_cfg = _variant_cfg(cfg, args.variant)
        finalize_evaluation(
            variant_cfg,
            args.run_dir,
            eval_seeds=_parse_eval_seeds(args.eval_seeds),
            eval_split_ids=_parse_eval_split_ids(args.eval_split_id),
            variant=args.variant,
        )
    elif args.stage == "ablation-evaluate":
        if not args.selected_config or not args.variant or args.variant == "full":
            raise SystemExit("--selected-config and a non-full --variant are required")
        variant_cfg = _variant_cfg(cfg, args.variant)
        _ensure_protocol_codebook(variant_cfg, args.stage, args.root, args.run_dir, variant=args.variant)
        cfg["codebook_path"] = variant_cfg["codebook_path"]
        cfg.setdefault("data", {})["codebook_path"] = variant_cfg["codebook_path"]
        run_evaluation(
            cfg,
            args.root,
            args.run_dir,
            args.selected_config,
            eval_seeds=_parse_eval_seeds(args.eval_seeds),
            eval_split_ids=_parse_eval_split_ids(args.eval_split_id),
            variant=args.variant,
            write_summary=not args.no_summary,
        )
    elif args.stage == "pair-results":
        if not args.comparison:
            raise SystemExit("--comparison is required for pair-results")
        split_artifact = args.split_artifact or str(Path(args.run_dir) / "d_eval_split_artifact.json")
        output_path = Path(args.paired_results) if args.paired_results else Path(args.run_dir) / "paired_results.json"
        result = build_paired_results(args.comparison, split_artifact, metric=args.metric)
        _write(output_path, result)
    elif args.stage == "analyze":
        if not args.paired_results:
            raise SystemExit("--paired-results is required for analyze")
        result = analyze_paired_results(args.paired_results, sesoi=args.sesoi)
        _write(Path(args.run_dir) / "statistical_analysis.json", result)


if __name__ == "__main__":
    main()
