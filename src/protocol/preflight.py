from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import load_json, validate_artifact
from .graph_config import load_graph_config


def run_preflight(run_dir: str, labels: np.ndarray, *, require_graph: bool = True) -> dict[str, Any]:
    run_path = Path(run_dir)
    eval_artifact = load_json(run_path / "d_eval_split_artifact.json")
    hpo_artifact = load_json(run_path / "d_hpo_split_artifact.json")
    validate_artifact(eval_artifact, labels, "d_eval")
    d_hpo_labels = np.asarray(labels)[np.asarray(eval_artifact["d_hpo_idx"], dtype=np.int64)]
    validate_artifact(hpo_artifact, d_hpo_labels, "d_hpo")

    graph = None
    if require_graph:
        graph = load_graph_config(str(run_path / "graph_config.json"), load_json(run_path / "edge_pilot.json"))

    report = {
        "ok": True,
        "dataset_size": int(len(labels)),
        "d_hpo_size": len(eval_artifact["d_hpo_idx"]),
        "d_eval_size": len(eval_artifact["d_eval_idx"]),
        "eval_split_count": len(eval_artifact["splits"]),
        "hpo_subset_fold_count": len(hpo_artifact["subset_folds"]),
        "hpo_full_fold_count": len(hpo_artifact["full_folds"]),
        "graph_config_fingerprint": graph["graph_config_fingerprint"] if graph else None,
    }
    (run_path / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
