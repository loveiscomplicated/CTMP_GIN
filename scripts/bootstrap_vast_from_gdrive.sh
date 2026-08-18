#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: bash scripts/bootstrap_vast_from_gdrive.sh <teds_file_id> <ce_run_file_id> <focal_run_file_id> [missing_corrected_file_id]"
  echo "Example: bash scripts/bootstrap_vast_from_gdrive.sh <TEDS_ID> <CE_RUN_ID> <FOCAL_RUN_ID>"
  exit 1
fi

TEDS_FILE_ID="$1"
CE_RUN_FILE_ID="$2"
FOCAL_RUN_FILE_ID="$3"
MISSING_CORRECTED_FILE_ID="${4:-}"

WORKSPACE_ROOT="/workspace"
REPO_URL="https://github.com/loveiscomplicated/CTMP_GIN.git"
REPO_DIR="${WORKSPACE_ROOT}/CTMP_GIN"
BRANCH="main"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_SPEC="${PROJECT_SPEC:-.[dev]}"

RAW_DATA_DIR="${REPO_DIR}/src/data/raw"
RUNS_DIR="${REPO_DIR}/runs"
DOWNLOAD_DIR="${WORKSPACE_ROOT}/downloads"

CE_RUN_ID="20260508-100129__los_ce_predictor__bs=1024__lr=1.00e-03__seed=1"
FOCAL_RUN_ID="20260508-152249__los_coarse_focal_sqrt_alpha_g1_predictor__bs=1024__lr=1.00e-03__seed=1"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

download_drive_file() {
  local file_id="$1"
  local output_path="$2"
  if [[ -f "$output_path" ]]; then
    echo "[$(ts)] already exists: $output_path"
    return 0
  fi
  echo "[$(ts)] downloading Google Drive file -> $output_path"
  "$PYTHON_BIN" -m gdown "https://drive.google.com/uc?id=${file_id}" -O "$output_path"
}

extract_run_archive() {
  local archive_path="$1"
  local expected_run_id="$2"
  if [[ -d "${RUNS_DIR}/${expected_run_id}" ]]; then
    echo "[$(ts)] run already extracted: ${RUNS_DIR}/${expected_run_id}"
    return 0
  fi
  echo "[$(ts)] extracting ${archive_path} -> ${RUNS_DIR}"
  tar -xzf "$archive_path" -C "$RUNS_DIR"
}

mkdir -p "$WORKSPACE_ROOT" "$DOWNLOAD_DIR"
cd "$WORKSPACE_ROOT"

apt update
apt install -y git wget python3-pip python3-dev python-is-python3 build-essential

if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "[$(ts)] repo exists -> update"
  cd "$REPO_DIR"
  git fetch --all
else
  echo "[$(ts)] cloning repo"
  git clone "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi

git checkout "$BRANCH"
git pull origin "$BRANCH"

cd "$REPO_DIR"
PYTHON_BIN="$PYTHON_BIN" PROJECT_SPEC="$PROJECT_SPEC" bash scripts/install_pip.sh

mkdir -p "$RAW_DATA_DIR" "$RUNS_DIR"

download_drive_file "$TEDS_FILE_ID" "${DOWNLOAD_DIR}/TEDS_Discharge.csv"
cp "${DOWNLOAD_DIR}/TEDS_Discharge.csv" "${RAW_DATA_DIR}/TEDS_Discharge.csv"

if [[ -n "$MISSING_CORRECTED_FILE_ID" ]]; then
  download_drive_file "$MISSING_CORRECTED_FILE_ID" "${DOWNLOAD_DIR}/missing_corrected.csv"
  cp "${DOWNLOAD_DIR}/missing_corrected.csv" "${RAW_DATA_DIR}/missing_corrected.csv"
fi

download_drive_file "$CE_RUN_FILE_ID" "${DOWNLOAD_DIR}/${CE_RUN_ID}.tgz"
download_drive_file "$FOCAL_RUN_FILE_ID" "${DOWNLOAD_DIR}/${FOCAL_RUN_ID}.tgz"

extract_run_archive "${DOWNLOAD_DIR}/${CE_RUN_ID}.tgz" "$CE_RUN_ID"
extract_run_archive "${DOWNLOAD_DIR}/${FOCAL_RUN_ID}.tgz" "$FOCAL_RUN_ID"

echo "[$(ts)] bootstrap complete"
echo "[$(ts)] repo    : $REPO_DIR"
echo "[$(ts)] data    : ${RAW_DATA_DIR}/TEDS_Discharge.csv"
echo "[$(ts)] CE ckpt : ${RUNS_DIR}/${CE_RUN_ID}/checkpoints/best.pt"
echo "[$(ts)] focal   : ${RUNS_DIR}/${FOCAL_RUN_ID}/checkpoints/best.pt"
echo "[$(ts)] next    : cd ${REPO_DIR}"
