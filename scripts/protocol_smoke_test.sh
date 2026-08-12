#!/usr/bin/env bash
set -euo pipefail

ts() { date '+%Y-%m-%d %H:%M:%S'; }
pass() { echo "[$(ts)] SMOKE PASS: $*"; }
fail() { echo "[$(ts)] SMOKE FAIL: $*" >&2; exit 1; }
info() { echo "[$(ts)] SMOKE INFO: $*"; }

RUN_DIR="${RUN_DIR:-new_runs}"
ROOT="${ROOT:-src/data}"
CONFIG="${CONFIG:-configs/ctmp_gin.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CODEBOOK_PATH="${CODEBOOK_PATH:-${CODEBOOK:-}}"
PROTOCOL_OPTUNA_STORAGE="${PROTOCOL_OPTUNA_STORAGE:-${OPTUNA_STORAGE:-}}"

export PROTOCOL_OPTUNA_STORAGE
export RUN_DIR

if [[ -n "$CODEBOOK_PATH" && ! -f "$CODEBOOK_PATH" ]]; then
  fail "CODEBOOK_PATH does not exist: $CODEBOOK_PATH"
fi
[[ -n "$PROTOCOL_OPTUNA_STORAGE" ]] || fail "PROTOCOL_OPTUNA_STORAGE must be set to a PostgreSQL Optuna URL"
case "$PROTOCOL_OPTUNA_STORAGE" in
  postgresql://*|postgresql+*://*|postgres://*) ;;
  sqlite://*) fail "SQLite storage is not allowed for this smoke; use PostgreSQL" ;;
  *) fail "PROTOCOL_OPTUNA_STORAGE must be PostgreSQL, got: $PROTOCOL_OPTUNA_STORAGE" ;;
esac

[[ -f "$CONFIG" ]] || fail "CONFIG does not exist: $CONFIG"
[[ -d "$ROOT/raw" ]] || fail "ROOT/raw does not exist: $ROOT/raw"

info "run_dir=$RUN_DIR"
info "root=$ROOT"
info "config=$CONFIG"
if [[ -n "$CODEBOOK_PATH" ]]; then
  info "codebook=$CODEBOOK_PATH"
else
  info "codebook=auto-generated at ${RUN_DIR}/codebook.json"
fi

info "running pytest"
"$PYTHON_BIN" -m pytest tests/ -q
pass "pytest passed"

info "checking PostgreSQL Optuna storage"
"$PYTHON_BIN" - <<'PY'
import os
import time
import optuna

storage = os.environ["PROTOCOL_OPTUNA_STORAGE"]
study_name = f"protocol_smoke_{int(time.time())}"
study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    direction="maximize",
    load_if_exists=False,
)
trial = study.ask()
study.tell(trial, 0.0)
print(f"SMOKE PASS: PostgreSQL Optuna storage reachable; study={study_name}")
PY

info "running protocol prepare"
PREPARE_CMD=(
  "$PYTHON_BIN" -m src.protocol.runner
  --stage prepare
  --config "$CONFIG"
  --root "$ROOT"
  --run-dir "$RUN_DIR"
)
if [[ -n "$CODEBOOK_PATH" ]]; then
  PREPARE_CMD+=(--codebook "$CODEBOOK_PATH")
fi
"${PREPARE_CMD[@]}"
pass "prepare completed"

info "running protocol preflight"
PREFLIGHT_CMD=(
  "$PYTHON_BIN" -m src.protocol.runner
  --stage preflight
  --config "$CONFIG"
  --root "$ROOT"
  --run-dir "$RUN_DIR"
)
if [[ -n "$CODEBOOK_PATH" ]]; then
  PREFLIGHT_CMD+=(--codebook "$CODEBOOK_PATH")
fi
"${PREFLIGHT_CMD[@]}"
pass "preflight completed"

info "checking real split sizes and paired-statistics path"
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from src.protocol.analysis import analyze_paired_results, build_paired_results

run_dir = Path(os.environ["RUN_DIR"])
split_path = run_dir / "d_eval_split_artifact.json"
artifact = json.loads(split_path.read_text(encoding="utf-8"))
splits = artifact["splits"]
if len(splits) != 15:
    raise SystemExit(f"expected 15 eval splits, got {len(splits)}")
if set(artifact["d_hpo_idx"]) & set(artifact["d_eval_idx"]):
    raise SystemExit("D_hpo and D_eval overlap")

candidate = {
    "graph_config_fingerprint": "smoke_graph",
    "results": [
        {"split_id": split["split_id"], "result": {"test_auc": 0.70}}
        for split in splits
    ],
}
reference = {
    "graph_config_fingerprint": "smoke_graph",
    "results": [
        {"split_id": split["split_id"], "result": {"test_auc": 0.69}}
        for split in splits
    ],
}
candidate_path = run_dir / "smoke_candidate_summary.json"
reference_path = run_dir / "smoke_reference_summary.json"
paired_path = run_dir / "smoke_paired_results.json"
candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
reference_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")

paired = build_paired_results(
    [f"F1,smoke_candidate,smoke_reference,{candidate_path},{reference_path}"],
    split_path,
)
paired_path.write_text(json.dumps(paired, indent=2), encoding="utf-8")
analysis = analyze_paired_results(str(paired_path))
comparison = paired["comparisons"][0]
raw = analysis["comparisons"][0]["raw"]
print("SMOKE PASS: split/pair/analyze path passed")
print(f"SMOKE INFO: split_sizes_constant={comparison['split_sizes_constant']}")
print(f"SMOKE INFO: n_train_values={sorted(set(comparison['n_train_values']))}")
print(f"SMOKE INFO: n_test_values={sorted(set(comparison['n_test_values']))}")
print(f"SMOKE INFO: nb_test_train_ratio={raw['test_train_ratio']}")
PY

pass "protocol smoke completed"
