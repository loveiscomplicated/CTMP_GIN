from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_GRAPH_FIELDS = (
    "score_method",
    "threshold",
    "top_k",
    "pruning_ratio",
    "estimator_version",
    "pilot_artifact_fingerprint",
    "source_model_name",
    "compatible_model_names",
)

DEFAULT_COMPATIBLE_MODEL_NAMES = (
    "ctmp_gin",
    "gin",
    "a3tgcn",
    "a3tgcn_2_points",
    "gin_gru",
    "gin_gru_2_points",
    "mlp",
    "xgboost",
)


def hub_concentration(edge_index, node_cardinalities: list[int]) -> float:
    """Fraction of edges incident to the highest-cardinality quartile of nodes."""
    import numpy as np

    if edge_index is None or edge_index.numel() == 0 or not node_cardinalities:
        return 0.0
    num_nodes = len(node_cardinalities)
    cutoff = max(1, int(np.ceil(num_nodes * 0.25)))
    hubs = set(np.argsort(np.asarray(node_cardinalities))[-cutoff:].tolist())
    edges = edge_index.detach().cpu().numpy()
    endpoints = np.mod(edges, num_nodes)
    incident = np.isin(endpoints, list(hubs)).any(axis=0)
    return float(incident.mean()) if incident.size else 0.0


def _normalise_compatible_models(values: dict[str, Any], source_model_name: str) -> list[str]:
    raw = values.get("compatible_model_names") or DEFAULT_COMPATIBLE_MODEL_NAMES
    models = sorted({str(model) for model in raw})
    if source_model_name not in models:
        models.append(source_model_name)
        models.sort()
    return models


def write_graph_config(
    path: str,
    values: dict[str, Any],
    pilot_artifact: dict[str, Any],
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    source_model_name = str(
        model_name
        or values.get("source_model_name")
        or pilot_artifact.get("model_name")
        or ""
    )
    if not source_model_name:
        raise ValueError("graph_config requires source_model_name/model_name")
    payload = {
        "score_method": values["score_method"],
        "threshold": float(values["threshold"]),
        "top_k": int(values["top_k"]),
        "pruning_ratio": float(values["pruning_ratio"]),
        "estimator_version": values.get("estimator_version", "categorical_plugin_v1"),
        "pilot_study": values.get("pilot_study"),
        "pilot_artifact_fingerprint": pilot_artifact["artifact_fingerprint"],
        "source_model_name": source_model_name,
        "compatible_model_names": _normalise_compatible_models(values, source_model_name),
    }
    payload["graph_config_fingerprint"] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=12
    ).hexdigest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_graph_config(
    path: str,
    pilot_artifact: dict[str, Any] | None = None,
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"required graph_config.json is missing: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_GRAPH_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"graph_config.json missing fields: {missing}")
    if payload["score_method"] not in {"raw_mi", "nmi"}:
        raise ValueError("graph_config.score_method must be raw_mi or nmi")
    compatible = payload.get("compatible_model_names")
    if not isinstance(compatible, list) or not all(isinstance(model, str) for model in compatible):
        raise ValueError("graph_config.compatible_model_names must be a list of model names")
    if payload["source_model_name"] not in compatible:
        raise ValueError("graph_config.source_model_name must be listed in compatible_model_names")
    if model_name is not None and model_name not in set(compatible):
        raise ValueError(
            f"graph_config is not marked compatible with model {model_name!r}; "
            f"compatible models: {compatible}"
        )
    if pilot_artifact is not None and payload["pilot_artifact_fingerprint"] != pilot_artifact.get("artifact_fingerprint"):
        raise ValueError("graph_config does not match the pilot artifact")
    return payload
