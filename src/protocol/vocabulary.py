from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


IGNORED_CODEBOOK_COLUMNS = {"DISYR", "CASEID", "REASON", "REASONb"}


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


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, float) and item.is_integer():
        return int(item)
    return item


def _sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int | float):
        return (1, float(value))
    if value is None:
        return (2, "")
    return (3, str(value))


def build_codebook_from_frame(frame: pd.DataFrame) -> dict[str, list[Any]]:
    codebook: dict[str, list[Any]] = {}
    for column in frame.columns:
        if column in IGNORED_CODEBOOK_COLUMNS:
            continue
        values = [_json_scalar(value) for value in frame[column].dropna().unique().tolist()]
        codebook[str(column)] = sorted(values, key=_sort_key)
    return codebook


def write_codebook_from_frame(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    source_csv: str | Path | None = None,
) -> dict[str, Any]:
    codebook = build_codebook_from_frame(frame)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(codebook, ensure_ascii=False, indent=2), encoding="utf-8")
    report: dict[str, Any] = {
        "path": str(target.resolve()),
        "feature_columns": len(codebook),
    }
    if source_csv is not None:
        report["source_csv"] = str(Path(source_csv).resolve())
    return report


def write_codebook_from_csv(
    source_csv: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    source = Path(source_csv)
    if not source.exists():
        raise FileNotFoundError(f"codebook source CSV does not exist: {source}")
    frame = pd.read_csv(source)
    return write_codebook_from_frame(frame, output_path, source_csv=source)


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
