from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stats import bh_fdr_adjust, holm_adjust, nadeau_bengio_corrected_t, tost


FAMILY_POLICY = {
    "F1": {"name": "primary CTMP-GIN vs A3T-GCN", "adjustment": "none", "max_comparisons": 1},
    "F2": {"name": "secondary baseline comparisons", "adjustment": "holm", "max_comparisons": None},
    "F3": {"name": "ablation superiority/degradation", "adjustment": "bh_fdr", "max_comparisons": None},
    "F4": {"name": "ablation equivalence", "adjustment": "bh_fdr", "max_comparisons": None},
}

ADJUSTERS = {
    "holm": holm_adjust,
    "bh_fdr": bh_fdr_adjust,
}


def _load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_from_result(result: dict[str, Any], metric: str) -> float:
    if metric in result:
        return float(result[metric])
    nested = result.get("result", {})
    if metric in nested:
        return float(nested[metric])
    raise KeyError(f"metric {metric!r} not found in evaluation result")


def _parse_comparison_spec(spec: str) -> dict[str, str]:
    parts = spec.split(",", 4)
    if len(parts) != 5:
        raise ValueError(
            "--comparison must use 'family,candidate,reference,candidate_summary,reference_summary'"
        )
    family, candidate, reference, candidate_summary, reference_summary = [part.strip() for part in parts]
    return {
        "family": family,
        "candidate": candidate,
        "reference": reference,
        "candidate_summary": candidate_summary,
        "reference_summary": reference_summary,
    }


def _unique_splits_by_id(splits: list[dict[str, Any]], *, context: str) -> dict[str, dict[str, Any]]:
    by_split: dict[str, dict[str, Any]] = {}
    for split in splits:
        split_id = split.get("split_id")
        if not split_id:
            raise ValueError(f"{context} contains a split without split_id")
        if split_id in by_split:
            raise ValueError(f"{context} contains duplicate split_id: {split_id}")
        by_split[str(split_id)] = split
    return by_split


def _unique_results_by_split_id(results: list[dict[str, Any]], *, context: str) -> dict[str, dict[str, Any]]:
    by_split: dict[str, dict[str, Any]] = {}
    for item in results:
        split_id = item.get("split_id")
        if not split_id:
            raise ValueError(f"{context} contains a result without split_id")
        if split_id in by_split:
            raise ValueError(f"{context} contains duplicate split_id: {split_id}")
        by_split[str(split_id)] = item["result"]
    return by_split


def _graph_fingerprints(candidate_summary: dict[str, Any], reference_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_graph_config_fingerprint": candidate_summary.get("graph_config_fingerprint"),
        "reference_graph_config_fingerprint": reference_summary.get("graph_config_fingerprint"),
        "same_graph_config": (
            candidate_summary.get("graph_config_fingerprint") is not None
            and candidate_summary.get("graph_config_fingerprint") == reference_summary.get("graph_config_fingerprint")
        ),
    }


def _split_size_summary(
    split_meta: dict[str, dict[str, Any]],
    split_ids: list[str],
) -> dict[str, Any]:
    n_train_values = [
        len(split_meta[split_id]["train_idx"]) + len(split_meta[split_id].get("val_idx", []))
        for split_id in split_ids
    ]
    n_test_values = [len(split_meta[split_id]["test_idx"]) for split_id in split_ids]
    ratios = [float(n_test) / float(n_train) for n_train, n_test in zip(n_train_values, n_test_values)]
    return {
        "n_train": int(round(sum(n_train_values) / len(n_train_values))),
        "n_test": int(round(sum(n_test_values) / len(n_test_values))),
        "n_train_values": n_train_values,
        "n_test_values": n_test_values,
        "nb_test_train_ratios": ratios,
        "nb_test_train_ratio": float(sum(ratios) / len(ratios)),
        "split_sizes_constant": len(set(n_train_values)) == 1 and len(set(n_test_values)) == 1,
    }


def build_paired_results(
    comparison_specs: list[str | dict[str, str]],
    split_artifact_path: str | Path,
    *,
    metric: str = "test_auc",
) -> dict[str, Any]:
    split_artifact = _load_summary(split_artifact_path)
    split_meta = _unique_splits_by_id(split_artifact.get("splits", []), context="split artifact")
    if not split_meta:
        raise ValueError("split artifact contains no evaluation splits")

    comparisons = []
    for raw_spec in comparison_specs:
        spec = _parse_comparison_spec(raw_spec) if isinstance(raw_spec, str) else raw_spec
        candidate_summary = _load_summary(spec["candidate_summary"])
        reference_summary = _load_summary(spec["reference_summary"])
        graph_metadata = _graph_fingerprints(candidate_summary, reference_summary)
        candidate_by_split = _unique_results_by_split_id(
            candidate_summary.get("results", []),
            context=f"{spec.get('candidate')} summary",
        )
        reference_by_split = _unique_results_by_split_id(
            reference_summary.get("results", []),
            context=f"{spec.get('reference')} summary",
        )
        candidate_split_ids = set(candidate_by_split)
        reference_split_ids = set(reference_by_split)
        if candidate_split_ids != reference_split_ids:
            raise ValueError(
                f"unpaired split sets for {spec.get('candidate')} vs {spec.get('reference')}: "
                f"candidate_only={sorted(candidate_split_ids - reference_split_ids)}, "
                f"reference_only={sorted(reference_split_ids - candidate_split_ids)}"
            )
        unknown_split_ids = sorted(candidate_split_ids - set(split_meta))
        if unknown_split_ids:
            raise ValueError(f"result summaries contain split_ids absent from split artifact: {unknown_split_ids}")
        split_ids = [split_id for split_id in split_meta if split_id in candidate_by_split]
        if not split_ids:
            raise ValueError(
                f"no paired splits for {spec.get('candidate')} vs {spec.get('reference')}"
            )

        candidate_values = [_metric_from_result(candidate_by_split[split_id], metric) for split_id in split_ids]
        reference_values = [_metric_from_result(reference_by_split[split_id], metric) for split_id in split_ids]
        differences = [candidate - reference for candidate, reference in zip(candidate_values, reference_values)]
        split_size_summary = _split_size_summary(split_meta, split_ids)
        comparisons.append({
            "family": spec["family"],
            "candidate": spec["candidate"],
            "reference": spec["reference"],
            "metric": metric,
            **graph_metadata,
            "split_ids": split_ids,
            "split_keys": [
                {
                    "split_id": split_id,
                    "eval_seed": split_meta[split_id].get("eval_seed"),
                    "fold": split_meta[split_id].get("fold"),
                }
                for split_id in split_ids
            ],
            "candidate_values": candidate_values,
            "reference_values": reference_values,
            "differences": differences,
            **split_size_summary,
        })

    return {
        "metric": metric,
        "split_artifact": str(Path(split_artifact_path)),
        "comparisons": comparisons,
    }


def analyze_paired_results(path: str, *, sesoi: float | None = None) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("comparisons", [])
    unknown_families = {record.get("family") for record in records} - set(FAMILY_POLICY)
    if unknown_families:
        raise ValueError(f"unknown comparison families: {sorted(map(str, unknown_families))}")
    for family, policy in FAMILY_POLICY.items():
        max_comparisons = policy["max_comparisons"]
        if max_comparisons is not None:
            count = sum(record.get("family") == family for record in records)
            if count > int(max_comparisons):
                raise ValueError(f"{family} allows at most {max_comparisons} comparison(s), got {count}")
    if any(record.get("family") == "F4" for record in records) and sesoi is None:
        raise ValueError("--sesoi is required for F4/TOST analysis")

    output = []
    for record in records:
        family = record["family"]
        policy = FAMILY_POLICY[family]
        result = {
            "family": family,
            "family_name": policy["name"],
            "adjustment": policy["adjustment"],
            "candidate": record.get("candidate"),
            "reference": record.get("reference"),
            "raw": nadeau_bengio_corrected_t(
                record["differences"],
                n_train=int(record["n_train"]) if "n_train" in record else None,
                n_test=int(record["n_test"]) if "n_test" in record else None,
                test_train_ratio=(
                    float(record["nb_test_train_ratio"])
                    if "nb_test_train_ratio" in record
                    else None
                ),
            ),
        }
        if family == "F4":
            result["tost"] = tost(record["differences"], float(sesoi))
        output.append(result)

    for family, policy in FAMILY_POLICY.items():
        adjustment = policy["adjustment"]
        if adjustment == "none":
            continue
        adjuster = ADJUSTERS[adjustment]
        indices = [i for i, result in enumerate(output) if result["family"] == family]
        adjusted = adjuster([output[i]["raw"]["p_value"] for i in indices]) if indices else []
        for index, value in zip(indices, adjusted):
            output[index]["adjusted_p_value"] = value
    return {"sesoi": sesoi, "family_policy": FAMILY_POLICY, "comparisons": output}
