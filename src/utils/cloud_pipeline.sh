#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/CTMP_GIN"
RUNS_DIR="/workspace/CTMP_GIN/runs"

cd "$REPO_DIR"

echo "===== pipeline start ====="

# --------------------------------------------------
# conda activate
# --------------------------------------------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pyg_2

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

# --------------------------------------------------
# stop runpod
# --------------------------------------------------
echo "===== stopping pod ====="

python src/utils/cloud_stop_pod.py

echo "===== pipeline complete ====="