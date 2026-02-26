#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Args
# -----------------------
if [[ $# -lt 2 ]]; then
  echo "Usage: bash run.sh <model_name> <config_path>"
  echo "Example: bash run.sh gin configs/gin.yaml"
  exit 1
fi

MODEL_NAME="$1"
CONFIG_PATH="$2"

echo "model_name: ${MODEL_NAME}"
echo "config    : ${CONFIG_PATH}"

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
mkdir -p /root/.config/rclone
echo "$RCLONE_CONF_B64" | base64 -d > /root/.config/rclone/rclone.conf
PIPELINE="$(cat <<'BASH'
set -euo pipefail
ts() { date '+%Y-%m-%d %H:%M:%S'; }

MODEL_NAME="__MODEL_NAME__"
CONFIG_PATH="__CONFIG_PATH__"

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

echo "[$(ts)] ===== pipeline start ====="
echo "[$(ts)] model_name: $MODEL_NAME"

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
# uv + venv
# -----------------------
# uv 설치 (standalone installer)
# - CI/컨테이너에서는 UV_UNMANAGED_INSTALL로 경로 고정 추천
#   (설치 스크립트가 쉘 프로필을 건드리지 않게)  :contentReference[oaicite:0]{index=0}
if ! command -v uv >/dev/null 2>&1; then
  echo "[$(ts)] installing uv"
  apt update
  apt install -y curl ca-certificates
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh
fi
echo "[$(ts)] uv: $(command -v uv)"
uv --version

# venv 생성 (python 3.12)
# uv는 .venv를 기본으로 사용하고, 필요하면 Python도 내려받을 수 있음 :contentReference[oaicite:1]{index=1}
cd "$REPO_DIR"
uv venv --python 3.12

# 활성화(이후 python/pip 대신 uv pip 써도 되지만,
# notify()에서 python을 쓰니 활성화해두면 안전)
source .venv/bin/activate

# -----------------------
# Python deps
# -----------------------
# 1) torch/torchvision
# PyTorch는 가속기별 빌드/별도 인덱스가 흔함(예: cu121, cpu 등) :contentReference[oaicite:2]{index=2}
# - 가장 단순: 기본 인덱스로 설치
# uv pip install torch torchvision

# (옵션) CUDA 인덱스를 명시하고 싶으면 예시:
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) torch_geometric
# PyG는 최소 설치는 `torch_geometric`만으로 가능(추가 확장 라이브러리는 선택) :contentReference[oaicite:3]{index=3}
uv pip install torch_geometric

# (옵션) 확장 라이브러리까지(휠 권장, torch/cuda 조합에 맞춰 data.pyg.org 사용) :contentReference[oaicite:4]{index=4}
# TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
# CUDA=$(python -c "import torch; print('cpu' if torch.version.cuda is None else 'cu'+torch.version.cuda.replace('.',''))")
# uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
#   -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"

# 3) 프로젝트 requirements
uv pip install -r requirements.txt
uv pip install requests

# -----------------------
# Data download
# -----------------------
cd "$DATA_DIR"
gdown "$GDOWN_FILE_ID"

# -----------------------
# Training
# -----------------------
cd "$REPO_DIR"
echo "[$(ts)] training start"
set +e
python -m src.trainers.runpod_optuna --config "$CONFIG_PATH"
TRAIN_RC=$?
set -e

if [[ $TRAIN_RC -eq 0 ]]; then
  notify "[SUCCESS] Training completed. model=$MODEL_NAME config=$CONFIG_PATH"
else
  notify "[FAIL] Training failed (rc=$TRAIN_RC). model=$MODEL_NAME config=$CONFIG_PATH"
fi

# -----------------------
# Upload runs (always try) + retry policy C
#   - upload fails => notify + HOLD (no shutdown)
# -----------------------
mkdir -p /root/.config/rclone
if [[ -z "${RCLONE_CONF_B64:-}" ]]; then
  notify "[UPLOAD_FAIL] RCLONE_CONF_B64 not set. Holding without shutdown."
  hold_forever
fi

echo "$RCLONE_CONF_B64" | base64 -d > /root/.config/rclone/rclone.conf

attempt=1
ok=0
while [[ $attempt -le $UPLOAD_RETRIES ]]; do
  echo "[$(ts)] upload attempt $attempt/$UPLOAD_RETRIES ..."
  if rclone copy "$RUNS_DIR" "${RCLONE_REMOTE}:${RCLONE_DEST_DIR}" \
      --create-empty-src-dirs \
      --transfers 8 \
      --checkers 16 \
      --retries 3 \
      --low-level-retries 10 \
      --stats 10s
  then
    ok=1
    break
  fi
  attempt=$((attempt+1))
  sleep 5
done

if [[ $ok -eq 1 ]]; then
  notify "[SUCCESS] Upload completed: ${RCLONE_REMOTE}:${RCLONE_DEST_DIR}"
  echo "[$(ts)] shutting down..."
  # -----------------------
  # Stop/Terminate pod (RunPod-native)
  # -----------------------
  echo "[$(ts)] stopping pod via runpodctl..."

  if command -v runpodctl >/dev/null 2>&1 && [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    # 1) stop (보통 과금 멈추는 목적이면 이걸 우선)
    runpodctl stop pod "$RUNPOD_POD_ID" && exit 0

    # 2) stop이 안 되면 remove(terminate) 시도
    runpodctl remove pod "$RUNPOD_POD_ID" && exit 0

    echo "[$(ts)] runpodctl stop/remove failed; falling back to process exit."
  fi

  # fallback: 컨테이너 프로세스 종료 (환경에 따라 pod가 내려갈 수도/아닐 수도)
  kill -TERM 1 || true
  exit 0
else
  notify "[UPLOAD_FAIL] Upload failed after ${UPLOAD_RETRIES} attempts. Holding without shutdown."
  hold_forever
fi
BASH
)"

# Fill placeholders
PIPELINE="${PIPELINE//__MODEL_NAME__/${MODEL_NAME}}"
PIPELINE="${PIPELINE//__CONFIG_PATH__/${CONFIG_PATH}}"
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

# -----------------------
# tmux session: create and start
# -----------------------
# Ensure tmux exists BEFORE using it
apt update
apt install -y tmux

if tmux has-session -t "${MODEL_NAME}" 2>/dev/null; then
  echo "[$(ts)] tmux session exists: ${MODEL_NAME}"
else
  echo "[$(ts)] creating tmux session: ${MODEL_NAME}"
  tmux new-session -d -s "${MODEL_NAME}"
fi

PIPE_PATH="/tmp/${MODEL_NAME}__pipeline.sh"
printf "%s" "$PIPELINE" > "$PIPE_PATH"
chmod +x "$PIPE_PATH"

# Run pipeline in that tmux session
tmux send-keys -t "${MODEL_NAME}" "bash $PIPE_PATH" C-m

echo "[$(ts)] started in tmux session '${MODEL_NAME}'."
echo "Attach with: tmux attach -t ${MODEL_NAME}"