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
BOT_NAME="runpod_setup"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

notify() {
  local msg="$1"
  if [[ -f "$SEND_MESSAGE_PY" ]]; then
    python "$SEND_MESSAGE_PY" "$msg" "$BOT_NAME" || true
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

# Load conda
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
else
    echo "[$(ts)] Error: $CONDA_SH not found"
    exit 1
fi

# ----------------------------------
# Accept Anaconda ToS (non-interactive fix)
# ----------------------------------
conda activate base || true

echo "[$(ts)] conda: $(command -v conda)"
conda --version

# ToS accept
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

conda config --set channels defaults || true
conda config --set channel_priority flexible || true

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[$(ts)] conda env $ENV_NAME exists -> skip create"
else
  echo "[$(ts)] creating conda env $ENV_NAME"
  conda create -y -n "$ENV_NAME" python=3.12
fi

# -----------------------
# Python deps (RTX 5090 / CUDA 12.8 대응 최적화)
# -----------------------
conda activate "$ENV_NAME"

# 1. pip 및 빌드 도구 최신화
python -m pip install -U pip setuptools wheel

# 2. PyTorch 설치 (RTX 5090 지원을 위해 가장 최신 인덱스 사용)
# 현재 5090은 매우 최신이므로 --pre(프리뷰) 버전을 사용하는 것이 가장 확실할 수 있습니다.
echo "[$(ts)] Installing PyTorch for RTX 5090..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# 3. 설치 확인 (매우 중요)
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')" || die "Torch check failed"

# 4. PyG 의존성 설치
# 5090 환경에서는 바이너리가 없을 확률이 높으므로 소스에서 빌드하도록 유도합니다.
echo "[$(ts)] Installing PyG dependencies (this may take a while)..."
pip install torch-scatter torch-sparse torch-cluster torch-spline_conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])").html

# 5. 나머지 설치
pip install torch-geometric
cd "$REPO_DIR"
pip install -r requirements.txt
pip install requests gdown

# -----------------------
# Data download
# -----------------------
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"
gdown "$GDOWN_FILE_ID"

echo "[$(ts)] Setup complete."
