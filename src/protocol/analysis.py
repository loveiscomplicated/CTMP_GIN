from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stats import bh_fdr_adjust, holm_adjust, nadeau_bengio_corrected_t, tost


def analyze_paired_results(path: str, *, sesoi: float | None = None) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("comparisons", [])
    if any(record.get("family") == "F4" for record in records) and sesoi is None:
        raise ValueError("--sesoi is required for F4/TOST analysis")

    output = []
    for record in records:
        family = record["family"]
        result = {
            "family": family,
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

    for family, adjuster in (("F2", holm_adjust), ("F3", bh_fdr_adjust), ("F4", bh_fdr_adjust)):
        indices = [i for i, result in enumerate(output) if result["family"] == family]
        adjusted = adjuster([output[i]["raw"]["p_value"] for i in indices]) if indices else []
        for index, value in zip(indices, adjusted):
            output[index]["adjusted_p_value"] = value
    return {"sesoi": sesoi, "comparisons": output}
