#!/usr/bin/env bash
set -uo pipefail  # set -e 제거: wait exit code가 set -e를 트리거하지 않도록

REPO_DIR="/workspace/CTMP_GIN"
RUNS_DIR="/workspace/CTMP_GIN/runs"

cd "$REPO_DIR"

# --------------------------------------------------
# conda 초기화
# --------------------------------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found"
    exit 1
fi

conda activate pyg_2

# 어떤 이유로든 스크립트가 종료될 때 pod는 반드시 멈추도록
trap 'echo "===== stopping pod ====="; python src/utils/cloud_stop_pod.py' EXIT

echo "===== parallel training start ====="

# -----------------------------
# run two experiments parallel
# -----------------------------
python -m src.main --config configs/ctmp_gin_remove_proj.yaml &
PID1=$!

python -m src.main --config configs/ctmp_gin_remove_gate.yaml &
PID2=$!

# -----------------------------
# wait for both, exit code 수집
# -----------------------------
FAIL1=0
FAIL2=0
wait $PID1 || FAIL1=$?
wait $PID2 || FAIL2=$?

if [ $FAIL1 -ne 0 ] || [ $FAIL2 -ne 0 ]; then
    echo "WARNING: one or more training runs failed (exit: $FAIL1, $FAIL2)"
    echo "Uploading whatever was saved before stopping..."
fi

echo "===== all training finished ====="

touch "$RUNS_DIR/DONE.ok"

# -----------------------------
# upload (trap이 pod stop을 처리하므로 업로드만)
# -----------------------------
bash src/utils/cloud_upload_drive.sh

echo "===== pipeline complete ====="
# EXIT trap이 pod stop 실행