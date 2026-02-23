#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/CTMP_GIN"
RUNS_DIR="/workspace/CTMP_GIN/runs"

cd "$REPO_DIR"

echo "===== pipeline start ====="

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

# --------------------------------------------------
# 어떤 이유로든 종료 시 pod stop 보장
# --------------------------------------------------
trap 'echo "===== stopping pod ====="; python src/utils/cloud_stop_pod.py' EXIT

# --------------------------------------------------
# train
# --------------------------------------------------
echo "===== training start ====="

python -m src.main --config configs/ctmp_gin_remove_proj.yaml
python -m src.main --config configs/ctmp_gin_remove_gate.yaml

echo "===== training finished ====="

touch "$RUNS_DIR/DONE.ok"

# --------------------------------------------------
# upload to Google Drive
# --------------------------------------------------
echo "===== upload start ====="

bash src/utils/cloud_upload_drive.sh

echo "===== upload finished ====="

echo "===== pipeline complete ====="
# EXIT trap이 pod stop 실행