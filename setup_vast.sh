#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Constants
# -----------------------
WORKSPACE_ROOT="/workspace"
REPO_URL="https://github.com/loveiscomplicated/CTMP_GIN.git"
REPO_DIR="${WORKSPACE_ROOT}/CTMP_GIN"
BRANCH="main"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_SPEC="${PROJECT_SPEC:-.[dev]}"

RUNS_DIR="${REPO_DIR}/runs"
DATA_DIR="${REPO_DIR}/src/data/raw"
GDOWN_FILE_ID="1T1oYAsdYDcdqUckd7CBzBWj9RnwGrEZg"

# rclone upload
RCLONE_REMOTE="gdrive"
RCLONE_DEST_DIR="CTMP_GIN_runs"
UPLOAD_RETRIES=3

# notifier
SEND_MESSAGE_PY="${REPO_DIR}/src/utils/send_message.py"
BOT_NAME="vast_setup"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

notify() {
  local msg="$1"
  if [[ -f "$SEND_MESSAGE_PY" ]]; then
    "$PYTHON_BIN" "$SEND_MESSAGE_PY" "$msg" "$BOT_NAME" || true
  else
    echo "[$(ts)] send_message.py not found: $SEND_MESSAGE_PY"
  fi
}

hold_forever() {
  echo "[$(ts)] holding forever..."
  while true; do sleep 3600; done
}

# -----------------------
# Environment Info
# -----------------------
echo "[$(ts)] VAST_CONTAINERLABEL='${VAST_CONTAINERLABEL:-}'"
echo "[$(ts)] VAST_INSTANCE_ID='${VAST_INSTANCE_ID:-}'"
if command -v vastai >/dev/null 2>&1; then
  echo "[$(ts)] vastai: $(command -v vastai)"
  vastai --version || true
else
  echo "[$(ts)] vastai CLI not found (will be installed at termination time)"
fi

# -----------------------
# System deps
# -----------------------
apt update
apt install -y tmux rclone git wget python3-pip python3-dev python-is-python3 build-essential

# tmux mouse
echo "set -g mouse on" >> ~/.tmux.conf || true
# Note: tmux source-file might fail if not inside a session, ignore error
tmux source-file ~/.tmux.conf || true

# -----------------------
# Repo setup
# -----------------------
mkdir -p "$WORKSPACE_ROOT"
cd "$WORKSPACE_ROOT"
if [[ -d "$REPO_DIR/.git" ]]; then
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

# -----------------------
# Python deps (pip, no conda)
# -----------------------
cd "$REPO_DIR"
PYTHON_BIN="$PYTHON_BIN" PROJECT_SPEC="$PROJECT_SPEC" TEDS_GDOWN_FILE_ID="$GDOWN_FILE_ID" DOWNLOAD_TEDS_DATA=1 bash scripts/install_pip.sh

echo "[$(ts)] Environment setup complete with pip."
# -----------------------
# Data download
# -----------------------
mkdir -p "$DATA_DIR"
if [[ -s "${DATA_DIR}/TEDS_Discharge.csv" ]]; then
  echo "[$(ts)] Data ready: ${DATA_DIR}/TEDS_Discharge.csv"
else
  cd "$DATA_DIR"
  "$PYTHON_BIN" -m gdown "https://drive.google.com/uc?id=${GDOWN_FILE_ID}" -O TEDS_Discharge.csv
fi

echo "[$(ts)] Setup complete."
