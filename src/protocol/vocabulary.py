from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_codebook(path: str) -> dict[str, list[Any]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"TEDS-D codebook does not exist: {target}")
    if target.suffix.lower() == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("codebook JSON must map column names to category lists")
        return {str(column): list(values) for column, values in payload.items()}
    if target.suffix.lower() in {".csv", ".tsv"}:
        frame = pd.read_csv(target, sep="\t" if target.suffix.lower() == ".tsv" else ",")
        required = {"column", "value"}
        if not required.issubset(frame.columns):
            raise ValueError("codebook CSV/TSV must contain 'column' and 'value' columns")
        return {
            str(column): group["value"].tolist()
            for column, group in frame.groupby("column", sort=False)
        }
    raise ValueError("codebook must be JSON, CSV, or TSV")


def encode_with_codebook(frame: pd.DataFrame, codebook: dict[str, list[Any]], *, oov_value: Any = None):
    encoded = frame.copy()
    oov_counts: dict[str, int] = {}
    for column in encoded.columns:
        if column in {"REASON", "REASONb"}:
            continue
        if column not in codebook:
            raise ValueError(f"codebook is missing feature column: {column}")
        mapping = {value: index for index, value in enumerate(codebook[column])}
        mapped = encoded[column].map(mapping)
        oov_counts[column] = int(mapped.isna().sum())
        # Use a valid, dedicated embedding row for OOV rather than a negative index.
        replacement = len(mapping) if oov_value is None else oov_value
        encoded[column] = mapped.fillna(replacement).astype(int)
    return encoded, oov_counts


def preflight_codebook(frame: pd.DataFrame, codebook_path: str) -> dict[str, Any]:
    codebook = load_codebook(codebook_path)
    feature_columns = [column for column in frame.columns if column not in {"REASON", "REASONb"}]
    missing = sorted(set(feature_columns) - set(codebook))
    if missing:
        raise ValueError(f"codebook missing columns: {missing}")
    _, oov_counts = encode_with_codebook(frame, codebook)
    return {
        "codebook_path": str(Path(codebook_path).resolve()),
        "feature_columns": len(feature_columns),
        "oov_counts": oov_counts,
        "oov_total": int(sum(oov_counts.values())),
    }
