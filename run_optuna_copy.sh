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

uv pip install dotenv
uv pip install optuna

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