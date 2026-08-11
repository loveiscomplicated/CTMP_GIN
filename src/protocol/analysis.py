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
                n_train=int(record["n_train"]),
                n_test=int(record["n_test"]),
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
