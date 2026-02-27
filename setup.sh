#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Constants
# -----------------------
WORKSPACE_ROOT="/workspace"
REPO_URL="https://github.com/loveiscomplicated/CTMP_GIN.git"
REPO_DIR="${WORKSPACE_ROOT}/CTMP_GIN"
BRANCH="runpod"

CONDA_DIR="$HOME/miniconda3"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
ENV_NAME="pyg_2"

RUNS_DIR="${REPO_DIR}/runs"
DATA_DIR="${REPO_DIR}/src/data/raw"
GDOWN_FILE_ID="1T1oYAsdYDcdqUckd7CBzBWj9RnwGrEZg"

# rclone upload
RCLONE_REMOTE="gdrive"
RCLONE_DEST_DIR="CTMP_GIN_runs"
UPLOAD_RETRIES=3

# notifier
SEND_MESSAGE_PY="${REPO_DIR}/src/utils/send_message.py"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# -----------------------
# Build pipeline to run INSIDE tmux
# -----------------------
PIPELINE="$(cat <<'BASH'
set -euo pipefail
ts() { date '+%Y-%m-%d %H:%M:%S'; }

WORKSPACE_ROOT="__WORKSPACE_ROOT__"
REPO_URL="__REPO_URL__"
REPO_DIR="__REPO_DIR__"
BRANCH="__BRANCH__"

CONDA_DIR="__CONDA_DIR__"
CONDA_SH="__CONDA_SH__"
ENV_NAME="__ENV_NAME__"

RUNS_DIR="__RUNS_DIR__"
DATA_DIR="__DATA_DIR__"
GDOWN_FILE_ID="__GDOWN_FILE_ID__"

RCLONE_REMOTE="__RCLONE_REMOTE__"
RCLONE_DEST_DIR="__RCLONE_DEST_DIR__"
UPLOAD_RETRIES="__UPLOAD_RETRIES__"

SEND_MESSAGE_PY="__SEND_MESSAGE_PY__"

notify() {
  local msg="$1"
  if [[ -f "$SEND_MESSAGE_PY" ]]; then
    python "$SEND_MESSAGE_PY" "$msg" || true
  else
    echo "[$(ts)] send_message.py not found: $SEND_MESSAGE_PY"
  fi
}

hold_forever() {
  echo "[$(ts)] holding forever..."
  while true; do sleep 3600; done
}

# 추가 (여기)
echo "[$(ts)] RUNPOD_POD_ID='${RUNPOD_POD_ID:-}'"
if command -v runpodctl >/dev/null 2>&1; then
  echo "[$(ts)] runpodctl: $(command -v runpodctl)"
  runpodctl --version || true
else
  echo "[$(ts)] runpodctl not found"
fi

# -----------------------
# System deps
# -----------------------
apt update
apt install -y tmux rclone git wget

# tmux mouse
echo "set -g mouse on" >> ~/.tmux.conf || true
tmux source-file ~/.tmux.conf || true

# -----------------------
# Repo setup
# -----------------------
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
# Miniconda + env
# -----------------------
cd "$WORKSPACE_ROOT"
if [[ ! -d "$CONDA_DIR" ]]; then
  echo "[$(ts)] installing miniconda"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_DIR"
else
  echo "[$(ts)] miniconda exists -> skip"
fi

source "$CONDA_SH"

# ----------------------------------
# Accept Anaconda ToS (non-interactive fix)
# ----------------------------------
# conda 함수 초기화 (tmux/non-interactive에서 중요)
conda activate base || true

# conda가 실제로 어디 걸리는지 로그로 확인
echo "[$(ts)] conda: $(command -v conda)"
conda --version

# ToS accept (base에 확실히 기록)
# conda 함수 초기화
conda activate base || true

# conda 경로 확인(진짜 실행 파일도 같이 보이게)
echo "[$(ts)] conda: $(type -a conda | head -n 2)"
conda --version

# ToS accept (에러 메시지에 나온 그대로)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# (선택) defaults 채널을 명시적으로 써서 override mismatch 방지
conda config --set channels defaults || true
conda config --set channel_priority flexible || true

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[$(ts)] conda env $ENV_NAME exists -> skip create"
else
  echo "[$(ts)] creating conda env $ENV_NAME"
  conda create -y -n "$ENV_NAME" python=3.12
fi

conda activate "$ENV_NAME"

# -----------------------
# Python deps (your order)
# -----------------------
python -m pip install -U pip
pip3 install torch torchvision
pip install torch_geometric
cd "$REPO_DIR"
pip install -r requirements.txt
pip install requests

# -----------------------
# Data download
# -----------------------
cd "$DATA_DIR"
gdown "$GDOWN_FILE_ID"

BASH
)"

# Fill placeholders
PIPELINE="${PIPELINE//__WORKSPACE_ROOT__/${WORKSPACE_ROOT}}"
PIPELINE="${PIPELINE//__REPO_URL__/${REPO_URL}}"
PIPELINE="${PIPELINE//__REPO_DIR__/${REPO_DIR}}"
PIPELINE="${PIPELINE//__BRANCH__/${BRANCH}}"
PIPELINE="${PIPELINE//__CONDA_DIR__/${CONDA_DIR}}"
PIPELINE="${PIPELINE//__CONDA_SH__/${CONDA_SH}}"
PIPELINE="${PIPELINE//__ENV_NAME__/${ENV_NAME}}"
PIPELINE="${PIPELINE//__RUNS_DIR__/${RUNS_DIR}}"
PIPELINE="${PIPELINE//__DATA_DIR__/${DATA_DIR}}"
PIPELINE="${PIPELINE//__GDOWN_FILE_ID__/${GDOWN_FILE_ID}}"
PIPELINE="${PIPELINE//__RCLONE_REMOTE__/${RCLONE_REMOTE}}"
PIPELINE="${PIPELINE//__RCLONE_DEST_DIR__/${RCLONE_DEST_DIR}}"
PIPELINE="${PIPELINE//__UPLOAD_RETRIES__/${UPLOAD_RETRIES}}"
PIPELINE="${PIPELINE//__SEND_MESSAGE_PY__/${SEND_MESSAGE_PY}}"