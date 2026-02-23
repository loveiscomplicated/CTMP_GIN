#!/usr/bin/env bash

set -e  # 에러 발생 시 즉시 종료

echo "===== RunPod setup start ====="
git checkout runpod

# --------------------------------------------------
# conda 초기화 (RunPod: /opt/conda, miniconda: $HOME/miniconda3)
# --------------------------------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found"
    exit 1
fi

# --------------------------------------------------
# create conda env
# --------------------------------------------------
if ! conda env list | grep -q pyg_2; then
    conda create -y -n pyg_2 python=3.12
fi

conda activate pyg_2

# --------------------------------------------------
# CUDA 버전 감지
# --------------------------------------------------
CUDA_TAG=$(nvcc --version 2>/dev/null | grep -oP 'release \K\d+\.\d+' | awk -F. '{printf "cu%s%s",$1,$2}')
CUDA_TAG=${CUDA_TAG:-cu121}
echo "Detected CUDA tag: $CUDA_TAG"

# --------------------------------------------------
# python packages
# --------------------------------------------------
pip install --upgrade pip
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "Installed PyTorch: $TORCH_VER"

pip install torch_geometric
pip install torch_scatter torch_sparse -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"

pip install -r requirements.txt
pip install requests

# --------------------------------------------------
# install tmux + rclone
# --------------------------------------------------
apt update
apt install -y tmux

curl -fsSL https://rclone.org/install.sh | bash
rclone version

# --------------------------------------------------
# data download
# --------------------------------------------------
cd src/data/raw

if [ ! -f "dataset_downloaded.flag" ]; then
    pip install gdown
    gdown 1T1oYAsdYDcdqUckd7CBzBWj9RnwGrEZg
    touch dataset_downloaded.flag
fi

cd /workspace/CTMP_GIN

echo "===== RunPod setup complete ====="